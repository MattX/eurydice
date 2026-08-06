//! A simple tree walking interpreter.

use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    ast::{
        self, BareListItem, EnumDefinition, Expression, FunctionDefinition, ListItem,
        PositionOrder, SetParam, Statement, StaticType, WithRange,
    },
    diagnostic::{
        DiagnosticLabel, DiagnosticSeverity, EngineDiagnostic, EvaluationFrame, LabelStyle,
        SourceId, SourceRange, TraceBinding, WarningKind, missing_return_warning, preview_value,
    },
    dice::{MultisetCrossProductIterator, Pool},
    error::{NonAdditiveSubject, ShapeMismatchError},
    operators::{apply_binary_op, apply_unary_op},
    primitives::{
        EXPLODE_ON_PRIMITIVE, EXPLODE_PRIMITIVE, Primitive, REROLL_ON_PRIMITIVE, REROLL_PRIMITIVE,
        register_primitives,
    },
    value::{display_requires_summing, sum_elements},
};

pub(crate) use crate::value::sum_pool;
pub use crate::{
    error::{ArityMismatch, RuntimeError, SemanticErrorKind},
    value::{ElementValue, RuntimeValue, SymbolTable},
};

/// Runtime bindings, organized as a stack of dynamically scoped function frames.
#[derive(Debug)]
struct ValEnv {
    frames: Vec<HashMap<String, RuntimeValue>>,
}

impl ValEnv {
    fn new() -> Self {
        Self {
            frames: vec![HashMap::new()],
        }
    }

    fn push_frame(&mut self) {
        self.frames.push(HashMap::new());
    }

    fn pop_frame(&mut self) {
        assert!(
            self.frames.len() > 1,
            "the global environment frame cannot be popped"
        );
        self.frames.pop();
    }

    fn get(&self, key: &str) -> Option<RuntimeValue> {
        self.frames
            .iter()
            .rev()
            .find_map(|frame| frame.get(key))
            .cloned()
    }

    fn insert(&mut self, key: String, value: RuntimeValue) {
        self.frames
            .last_mut()
            .expect("an environment always has a global frame")
            .insert(key, value);
    }

    fn contains(&self, key: &str) -> bool {
        self.frames
            .iter()
            .rev()
            .any(|frame| frame.contains_key(key))
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        self.frames
            .iter()
            .rev()
            .flat_map(|frame| frame.keys().map(String::as_str))
    }
}

#[derive(Debug, Clone)]
pub enum Function {
    Primitive(&'static Primitive),
    UserDefined(Rc<UserFunction>),
}

#[derive(Debug, Clone)]
pub struct UserFunction {
    definition: FunctionDefinition,
    arg_types: Vec<Option<StaticType>>,
    source_id: SourceId,
}

impl Function {
    /// The shape each argument is converted to, where the function names one.
    ///
    /// Shape is all a signature can ask for: `:n`, `:s` and `:d` say whether an
    /// argument arrives as a single value, a sequence or a die, and nothing
    /// says anything about what is inside it.
    fn get_arg_types(&self) -> Vec<Option<StaticType>> {
        match self {
            Function::Primitive(primitive) => primitive.arg_types.to_vec(),
            Function::UserDefined(function) => function.arg_types.clone(),
        }
    }
}

struct EvalContext {
    recursion_depth: usize,
    block_depth: usize,
    source_id: SourceId,
}

impl EvalContext {
    fn new(source_id: SourceId) -> Self {
        Self {
            recursion_depth: 0,
            block_depth: 0,
            source_id,
        }
    }

    fn nested_block(&self) -> Self {
        Self {
            recursion_depth: self.recursion_depth,
            block_depth: self.block_depth + 1,
            source_id: self.source_id,
        }
    }
}

/// What a name already refers to, when a binding tries to reuse it.
#[derive(Debug, Clone, Copy)]
enum EnumIdentifierKind {
    Set,
    Symbol,
}

impl std::fmt::Display for EnumIdentifierKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnumIdentifierKind::Set => write!(f, "symbol set"),
            EnumIdentifierKind::Symbol => write!(f, "symbol"),
        }
    }
}

pub struct Evaluator {
    env: ValEnv,
    outputs: Vec<EvaluatedOutput>,
    functions: HashMap<String, Function>,
    symbols: SymbolTable,
    /// Declared set names and symbol names, kept separately from the symbol
    /// table because they exist only to police the shared namespace.
    enums: HashSet<String>,
    enum_members: HashSet<String>,
    explode_depth: usize,
    recursion_depth: usize,
    lowest_first: bool,
    print_callback: Option<Box<dyn Fn(String, String)>>,
    source_id: SourceId,
    diagnostics: Vec<EngineDiagnostic>,
    /// Warnings already reported, so repeated evaluations of the same
    /// expression do not build a diagnostic only to discard it.
    warned: HashSet<(WarningKind, SourceRange)>,
}

/// A top-level output and its presentation metadata.
#[derive(Debug, Clone)]
pub struct EvaluatedOutput {
    pub value: RuntimeValue,
    pub name: String,
    pub field_names: Option<Vec<String>>,
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator {
    pub fn new() -> Self {
        let mut functions = HashMap::new();
        register_primitives(&mut functions);
        Self {
            env: ValEnv::new(),
            outputs: Vec::new(),
            functions,
            symbols: SymbolTable::default(),
            enums: HashSet::new(),
            enum_members: HashSet::new(),
            explode_depth: 2,
            recursion_depth: 10,
            lowest_first: false,
            print_callback: None,
            source_id: SourceId(0),
            diagnostics: Vec::new(),
            warned: HashSet::new(),
        }
    }

    pub fn begin_submission(&mut self, source_id: SourceId) {
        self.source_id = source_id;
        self.diagnostics.clear();
        self.warned.clear();
    }

    pub fn take_diagnostics(&mut self) -> Vec<EngineDiagnostic> {
        std::mem::take(&mut self.diagnostics)
    }

    /// The sources a diagnostic can still be asked to point into.
    ///
    /// Only function bodies outlive their submission: a diagnostic raised
    /// inside one is rendered against the source that defined it, however long
    /// ago that was. Everything else is unreachable once the submission ends.
    pub(crate) fn live_source_ids(&self) -> HashSet<SourceId> {
        self.functions
            .values()
            .filter_map(|function| match function {
                Function::UserDefined(function) => Some(function.source_id),
                Function::Primitive(_) => None,
            })
            .collect()
    }

    pub(crate) fn variable_suggestions(&self, name: &str) -> Vec<String> {
        closest_names(name, self.env.names())
    }

    pub(crate) fn function_suggestions(&self, name: &str) -> Vec<String> {
        closest_names(name, self.functions.keys().map(String::as_str))
    }

    /// Whether this warning has not been reported for this span yet.
    ///
    /// Callers check this before building a diagnostic: warnings are raised
    /// from inside evaluation loops, where the same expression can be visited
    /// once per outcome of a pool.
    fn should_warn(&mut self, kind: WarningKind, source: SourceId, range: ast::Range) -> bool {
        self.warned.insert((kind, SourceRange { source, range }))
    }

    /// `subject` is a plural phrase naming the two uses; it is always followed
    /// by "refer to the same pool".
    fn warn_pool_reuse(&mut self, source: SourceId, range: ast::Range, subject: &str) {
        const KIND: WarningKind = WarningKind::IndependentPoolReuse;
        if !self.should_warn(KIND, source, range) {
            return;
        }
        self.diagnostics.push(EngineDiagnostic {
            code: KIND.code().to_string(),
            severity: DiagnosticSeverity::Warning,
            summary: "The same dice pool is sampled independently more than once".to_string(),
            labels: vec![DiagnosticLabel {
                range: SourceRange { source, range },
                message: Some(format!("{subject} refer to the same pool")),
                style: LabelStyle::Primary,
            }],
            notes: vec![
                "Assigning a dice pool to a variable stores its distribution; it does not roll \
                 the dice and remember one result."
                    .to_string(),
            ],
            help: Some(
                "To reuse one roll, pass the pool to an `n` parameter and reuse that parameter \
                 inside the function."
                    .to_string(),
            ),
            fixes: Vec::new(),
            trace: Vec::new(),
            incomplete: false,
        });
    }

    fn warn_limit(
        &mut self,
        source: SourceId,
        range: ast::Range,
        kind: WarningKind,
        setting: &str,
        value: usize,
        summary: &str,
    ) {
        if !self.should_warn(kind, source, range) {
            return;
        }
        self.diagnostics.push(EngineDiagnostic {
            code: kind.code().to_string(),
            severity: DiagnosticSeverity::Warning,
            summary: summary.to_string(),
            labels: vec![DiagnosticLabel {
                range: SourceRange { source, range },
                message: Some(format!("`{setting}` is currently {value}")),
                style: LabelStyle::Primary,
            }],
            notes: vec!["The returned distribution is bounded by this setting.".to_string()],
            help: Some(format!(
                "Change it with `set \"{setting}\" to N` if you need a different bound."
            )),
            fixes: Vec::new(),
            trace: Vec::new(),
            incomplete: false,
        });
    }

    pub fn set_print_callback(&mut self, callback: Box<dyn Fn(String, String)>) {
        self.print_callback = Some(callback);
    }

    /// The declared symbols, needed to render any value that may contain one.
    pub fn symbols(&self) -> &SymbolTable {
        &self.symbols
    }

    pub fn execute(&mut self, statement: &WithRange<Statement>) -> Result<(), RuntimeError> {
        let eval_context = EvalContext::new(self.source_id);
        let result = self.execute_statement(&eval_context, statement)?;
        debug_assert!(
            result.is_none(),
            "value returned from a top-level statement"
        );
        Ok(())
    }

    pub fn take_outputs(&mut self) -> Vec<EvaluatedOutput> {
        std::mem::take(&mut self.outputs)
    }

    fn enum_identifier_kind(&self, name: &str) -> Option<EnumIdentifierKind> {
        if self.enums.contains(name) {
            Some(EnumIdentifierKind::Set)
        } else if self.enum_members.contains(name) {
            Some(EnumIdentifierKind::Symbol)
        } else {
            None
        }
    }

    /// Rejects a name that a declaration has already claimed.
    ///
    /// Assignments, loop variables and function parameters all bind a variable,
    /// and a declared name is unavailable to all three, so they report it the
    /// same way. Which one it was is shown by where the diagnostic points.
    fn validate_binding_name(&self, name: &str, range: ast::Range) -> Result<(), RuntimeError> {
        let Some(identifier_kind) = self.enum_identifier_kind(name) else {
            return Ok(());
        };
        Err(RuntimeError::Semantic {
            kind: SemanticErrorKind::BindingConflict,
            range: range.into(),
            message: format!(
                "`{name}` is a declared {identifier_kind}, so it cannot be used as a variable name"
            ),
        })
    }

    fn define_enum(
        &mut self,
        eval_context: &EvalContext,
        definition: &EnumDefinition,
        statement_range: ast::Range,
    ) -> Result<(), RuntimeError> {
        if eval_context.recursion_depth != 0 || eval_context.block_depth != 0 {
            return Err(RuntimeError::Semantic {
                kind: SemanticErrorKind::TopLevelOnly,
                range: statement_range.into(),
                message: "enum declarations are only allowed at the top level".to_string(),
            });
        }
        if let Some(kind) = self.enum_identifier_kind(&definition.name.value) {
            return Err(RuntimeError::Semantic {
                kind: SemanticErrorKind::BindingConflict,
                range: definition.name.range.into(),
                message: format!(
                    "enum {} conflicts with an existing enum {kind}",
                    definition.name.value
                ),
            });
        }
        if self.env.contains(&definition.name.value) {
            return Err(RuntimeError::Semantic {
                kind: SemanticErrorKind::BindingConflict,
                range: definition.name.range.into(),
                message: format!("{} is already bound as a variable", definition.name.value),
            });
        }

        let mut seen = HashSet::new();
        for member in &definition.members {
            if !seen.insert(member.value.as_str()) {
                return Err(RuntimeError::Semantic {
                    kind: SemanticErrorKind::BindingConflict,
                    range: member.range.into(),
                    message: format!("enum member {} is already defined", member.value),
                });
            }
            if member.value == definition.name.value {
                return Err(RuntimeError::Semantic {
                    kind: SemanticErrorKind::BindingConflict,
                    range: member.range.into(),
                    message: format!("enum member {} conflicts with its enum type", member.value),
                });
            }
            if let Some(kind) = self.enum_identifier_kind(&member.value) {
                return Err(RuntimeError::Semantic {
                    kind: SemanticErrorKind::BindingConflict,
                    range: member.range.into(),
                    message: format!(
                        "enum member {} conflicts with an existing enum {kind}",
                        member.value
                    ),
                });
            }
            if self.env.contains(&member.value) {
                return Err(RuntimeError::Semantic {
                    kind: SemanticErrorKind::BindingConflict,
                    range: member.range.into(),
                    message: format!("{} is already bound as a variable", member.value),
                });
            }
        }

        let member_names = definition
            .members
            .iter()
            .map(|member| member.value.clone())
            .collect::<Vec<_>>();
        let indices = self
            .symbols
            .define_set(definition.name.value.clone(), &member_names);
        for (member, symbol) in definition.members.iter().zip(indices) {
            self.env.insert(
                member.value.clone(),
                RuntimeValue::Element(ElementValue::Symbol(symbol)),
            );
            self.enum_members.insert(member.value.clone());
        }
        self.enums.insert(definition.name.value.clone());
        Ok(())
    }

    /// Executes a statement; returns a value if a `return` statement is encountered.
    fn execute_statement(
        &mut self,
        eval_context: &EvalContext,
        statement: &WithRange<Statement>,
    ) -> Result<Option<RuntimeValue>, RuntimeError> {
        match &statement.value {
            Statement::Assignment { name, value } => {
                self.validate_binding_name(&name.value, name.range)?;
                let value = self.evaluate(eval_context, value)?;
                self.env.insert(name.value.clone(), value);
            }
            Statement::FunctionDefinition(fd) => {
                if self.should_warn(
                    WarningKind::MissingResult,
                    eval_context.source_id,
                    fd.name.range,
                ) && let Some(warning) = missing_return_warning(eval_context.source_id, fd)
                {
                    self.diagnostics.push(warning);
                }
                for arg in &fd.args {
                    self.validate_binding_name(&arg.value.name, arg.range)?;
                }
                let arg_types = fd.args.iter().map(|arg| arg.value.ty).collect();
                self.functions.insert(
                    fd.name.value.clone(),
                    Function::UserDefined(Rc::new(UserFunction {
                        definition: fd.clone(),
                        arg_types,
                        source_id: eval_context.source_id,
                    })),
                );
            }
            Statement::EnumDefinition(definition) => {
                self.define_enum(eval_context, definition, statement.range)?;
            }
            Statement::Output {
                expr,
                named,
                labeled,
            } => {
                if eval_context.recursion_depth != 0 {
                    return Err(RuntimeError::OutputNotAtTopLevel {
                        range: statement.range.into(),
                    });
                }
                let value = self.evaluate(eval_context, expr)?;
                // `output` displays the summed pool, so reject one that cannot be
                // summed here: the conversion in `output.rs` runs after evaluation and
                // has no way to report an error.
                let summed = match &value {
                    RuntimeValue::Pool(pool) => Some(
                        sum_pool(pool)
                            .map_err(|error| error.into_error(expr.range, "displaying a pool"))?,
                    ),
                    _ => None,
                };
                let displayed: Vec<&ElementValue> = match (&value, &summed) {
                    (_, Some(pool)) => pool
                        .ordered_outcomes()
                        .iter()
                        .map(|(outcome, _)| outcome)
                        .collect(),
                    (RuntimeValue::List(values), None) => values.iter().collect(),
                    (RuntimeValue::Element(value), None) => vec![value],
                    (RuntimeValue::Pool(_), None) => {
                        unreachable!("a pool value is always summed above")
                    }
                };
                // Every outcome agrees on its shape from here on, so the first
                // one speaks for all of them.
                reject_mixed_output_fields(&displayed, expr.range)?;
                let field_names = if let Some(labels) = labeled {
                    let Some(ElementValue::Tuple(fields)) = displayed.first() else {
                        return Err(RuntimeError::LabelsOnNonTupleOutput {
                            range: labels.range.into(),
                            value_range: expr.range.into(),
                            value,
                        });
                    };
                    if labels.value.len() != fields.len() {
                        return Err(RuntimeError::OutputLabelCountMismatch {
                            range: labels.range.into(),
                            expected: fields.len(),
                            found: labels.value.len(),
                        });
                    }
                    Some(
                        labels
                            .value
                            .iter()
                            .map(|label| {
                                interpolate_variable_names(label, &self.env, &self.symbols)
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                } else {
                    None
                };
                let name = match named {
                    Some(name) => interpolate_variable_names(name, &self.env, &self.symbols)?,
                    None => format!("output {}", self.outputs.len() + 1),
                };
                // Hand on the pool already summed above, so that displaying it
                // does not repeat the summation.
                let value = match (&value, summed) {
                    (RuntimeValue::Pool(pool), Some(summed)) if display_requires_summing(pool) => {
                        RuntimeValue::Pool(Rc::new(summed))
                    }
                    _ => value,
                };
                self.outputs.push(EvaluatedOutput {
                    value,
                    name,
                    field_names,
                });
            }
            Statement::Print { expr, named } => {
                let value = self.evaluate(eval_context, expr)?;
                let name = match named {
                    Some(name) => interpolate_variable_names(name, &self.env, &self.symbols)?,
                    None => "".to_string(),
                };
                if let Some(ref callback) = self.print_callback {
                    callback(value.display(&self.symbols).to_string(), name);
                }
            }
            Statement::Return { value } => {
                if self.recursion_depth == 0 {
                    return Err(RuntimeError::ReturnOutsideFunction {
                        range: statement.range.into(),
                    });
                }
                let value = self.evaluate(eval_context, value)?;
                return Ok(Some(value));
            }
            Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                let condition_value = self.evaluate(eval_context, condition)?;
                let cond_value = match condition_value {
                    RuntimeValue::Element(ElementValue::AdditiveIdentity) => 0,
                    RuntimeValue::Element(ElementValue::Int(i)) => i,
                    _ => {
                        return Err(RuntimeError::InvalidCondition {
                            range: condition.range.into(),
                            value: condition_value,
                        });
                    }
                };
                let block = if cond_value != 0 {
                    then_block
                } else if let Some(block) = else_block {
                    block
                } else {
                    return Ok(None);
                };
                let nested_context = eval_context.nested_block();
                for statement in block {
                    let res = self.execute_statement(&nested_context, statement)?;
                    if res.is_some() {
                        return Ok(res);
                    }
                }
            }
            Statement::Loop {
                variable,
                range_expression,
                body,
            } => {
                self.validate_binding_name(&variable.value, variable.range)?;
                let range = self.evaluate(eval_context, range_expression)?;
                let range = match range {
                    RuntimeValue::List(range) => range,
                    _ => {
                        return Err(RuntimeError::LoopOverNonSequence {
                            range: range_expression.range.into(),
                            value: range,
                        });
                    }
                };
                let nested_context = eval_context.nested_block();
                for value in range.iter() {
                    self.env
                        .insert(variable.value.clone(), RuntimeValue::Element(value.clone()));
                    for statement in body {
                        let res = self.execute_statement(&nested_context, statement)?;
                        if res.is_some() {
                            return Ok(res);
                        }
                    }
                }
            }
            Statement::Set(s) => {
                if eval_context.recursion_depth != 0 {
                    return Err(RuntimeError::SetNotAtTopLevel {
                        range: statement.range.into(),
                    });
                }
                match s {
                    SetParam::PositionOrder(order) => {
                        self.lowest_first = *order == PositionOrder::Ascending
                    }
                    SetParam::ExplodeDepth(d) => self.explode_depth = *d,
                    SetParam::MaximumFunctionDepth(d) => self.recursion_depth = *d,
                }
            }
        }
        Ok(None)
    }

    fn evaluate(
        &mut self,
        eval_context: &EvalContext,
        expression: &WithRange<Expression>,
    ) -> Result<RuntimeValue, RuntimeError> {
        match &expression.value {
            Expression::UnaryOp { op, operand } => {
                let value = self.evaluate(eval_context, operand)?;
                apply_unary_op(op, &value)
            }
            Expression::BinaryOp { op, left, right } => {
                let left_value = self.evaluate(eval_context, left)?;
                let right_value = self.evaluate(eval_context, right)?;
                if same_pool(&left_value, &right_value) {
                    let subject = match (&left.value, &right.value) {
                        (Expression::Reference(left), Expression::Reference(right))
                            if left == right =>
                        {
                            format!("Both uses of `{left}`")
                        }
                        (Expression::Reference(left), Expression::Reference(right)) => {
                            format!("`{left}` and `{right}`")
                        }
                        _ => "Both operands".to_string(),
                    };
                    self.warn_pool_reuse(eval_context.source_id, op.range, &subject);
                }
                Ok(apply_binary_op(
                    op,
                    &left_value,
                    left.range,
                    &right_value,
                    self.lowest_first,
                )?)
            }
            Expression::List(list) => {
                let items = list
                    .items
                    .iter()
                    .map(|item| self.evaluate_list_literal_item(eval_context, item))
                    .collect::<Result<Vec<_>, _>>()?;
                // A sequence holds whatever it was written with; nothing here
                // asks the items to agree. An operation that later needs them
                // to is where a disagreement surfaces.
                let mut flattened = Vec::new();
                for (item, source) in items.iter().zip(list.items.iter()) {
                    flattened.extend(item.to_list(1).map_err(|mismatch| {
                        mismatch.into_error(
                            list_item_range(source),
                            "flattening a pool into a sequence",
                            NonAdditiveSubject::Sequence(item.elements().len()),
                        )
                    })?);
                }
                Ok(RuntimeValue::List(Rc::new(flattened)))
            }
            Expression::FunctionCall { name, args } => {
                let func = self
                    .functions
                    .get(&name.value)
                    .ok_or_else(|| RuntimeError::UndefinedFunction {
                        range: expression.range.into(),
                        name: name.value.clone(),
                        arity_mismatch: arity_mismatch(&self.functions, &name.value, args),
                    })?
                    .clone();
                let ranges = args.iter().map(|arg| arg.range).collect::<Vec<_>>();
                let args = args
                    .iter()
                    .map(|arg| self.evaluate(eval_context, arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let args = args
                    .into_iter()
                    .zip(ranges.iter())
                    .map(|(arg, range)| WithRange {
                        value: arg,
                        range: *range,
                    })
                    .collect::<Vec<_>>();
                let func_with_range = WithRange {
                    value: func,
                    range: name.range,
                };
                self.evaluate_function_call(eval_context, &func_with_range, args)
            }
            Expression::Reference(name) => {
                self.env
                    .get(name)
                    .ok_or_else(|| RuntimeError::UndefinedReference {
                        range: expression.range.into(),
                        name: name.clone(),
                    })
            }
            Expression::Int(i) => Ok(RuntimeValue::Element(ElementValue::Int(*i))),
        }
    }

    /// Evaluates a list literal item.
    ///
    /// This doesn't take precomputed values, because literals can contain special non-expression syntax.
    fn evaluate_list_literal_item(
        &mut self,
        eval_context: &EvalContext,
        item: &ListItem,
    ) -> Result<RuntimeValue, RuntimeError> {
        let repeat_count = match &item.repeat {
            Some(repeat) => match self.evaluate(eval_context, repeat)? {
                RuntimeValue::Element(ElementValue::AdditiveIdentity) => 0,
                RuntimeValue::Element(ElementValue::Int(i)) => {
                    usize::try_from(i.max(0)).expect("converting a positive i32 into usize")
                }
                repeat_value => {
                    return Err(RuntimeError::InvalidRepeatExpression {
                        range: repeat.range.into(),
                        value: repeat_value,
                    });
                }
            },
            None => 1,
        };
        let base = match &item.item {
            BareListItem::Expr(expression) => self.evaluate(eval_context, expression),
            BareListItem::Range(start_expr, end_expr) => {
                let start = self.evaluate(eval_context, start_expr)?;
                let start = match start {
                    RuntimeValue::Element(ElementValue::AdditiveIdentity) => 0,
                    RuntimeValue::Element(ElementValue::Int(i)) => i,
                    _ => {
                        return Err(RuntimeError::RangeHasNonSequenceEndpoints {
                            range: start_expr.range.into(),
                            value: start,
                        });
                    }
                };
                let end = self.evaluate(eval_context, end_expr)?;
                let end = match end {
                    RuntimeValue::Element(ElementValue::AdditiveIdentity) => 0,
                    RuntimeValue::Element(ElementValue::Int(i)) => i,
                    _ => {
                        return Err(RuntimeError::RangeHasNonSequenceEndpoints {
                            range: end_expr.range.into(),
                            value: end,
                        });
                    }
                };
                Ok(RuntimeValue::List(Rc::new(
                    (start..=end).map(ElementValue::Int).collect(),
                )))
            }
        }?;
        let item_range = match &item.item {
            BareListItem::Expr(expression) => expression.range,
            BareListItem::Range(start_expr, end_expr) => {
                (start_expr.range.start, end_expr.range.end).into()
            }
        };
        let elements = base.to_list(repeat_count).map_err(|mismatch| {
            mismatch.into_error(
                item_range,
                "flattening a pool into a sequence",
                NonAdditiveSubject::Sequence(base.elements().len()),
            )
        })?;
        Ok(RuntimeValue::List(Rc::new(elements)))
    }

    fn evaluate_function_call(
        &mut self,
        eval_context: &EvalContext,
        function: &WithRange<Function>,
        args: Vec<WithRange<RuntimeValue>>,
    ) -> Result<RuntimeValue, RuntimeError> {
        if eval_context.recursion_depth >= self.recursion_depth {
            self.warn_limit(
                eval_context.source_id,
                function.range,
                WarningKind::MaximumFunctionDepth,
                "maximum function depth",
                self.recursion_depth,
                "Maximum function depth stopped this call",
            );
            return Ok(RuntimeValue::empty_list());
        }

        // First, convert the arguments to the expected types.
        let expected_types = function.value.get_arg_types();
        if expected_types.len() != args.len() {
            panic!(
                "wrong number of arguments; this should have been caught by function name matching"
            );
        }
        for left_index in 0..args.len() {
            for right_index in left_index + 1..args.len() {
                let samples_left =
                    expected_types[left_index].is_some_and(|shape| shape != StaticType::Pool);
                let samples_right =
                    expected_types[right_index].is_some_and(|shape| shape != StaticType::Pool);
                if samples_left
                    && samples_right
                    && same_pool(&args[left_index].value, &args[right_index].value)
                {
                    self.warn_pool_reuse(
                        eval_context.source_id,
                        function.range,
                        &format!("Arguments {} and {}", left_index + 1, right_index + 1),
                    );
                }
            }
        }
        let arg_ranges = args.iter().map(|arg| arg.range).collect::<Vec<_>>();
        let args = args
            .into_iter()
            .zip(expected_types.iter())
            .map(|(arg, expected)| coerce_arg(arg.value, *expected, arg.range))
            .collect::<Result<Vec<_>, _>>()?;

        // This vector will contain references to pools which match an int or list arguments. These
        // are the pools over which we need to iterate to get the argument values.
        let mut pools = Vec::new();
        // Contains (index, is_element) for the corresponding pool.
        let mut pool_iterator_info = Vec::new();
        for (i, (arg, expected_type)) in args.iter().zip(expected_types.iter()).enumerate() {
            let pool = match arg {
                RuntimeValue::Pool(p) => Some(p),
                _ => None,
            };
            if let (Some(p), Some(shape)) = (pool, expected_type) {
                if !matches!(shape, StaticType::Int | StaticType::List) {
                    continue;
                }
                // If the expected type is an Int, `coerce_arg` has already turned the pool into a sum,
                // with outcomes of length 1.
                pools.push(p);
                pool_iterator_info.push((i, *shape == StaticType::Int));
            }
        }

        if pools.is_empty() {
            return self.call_function(eval_context, function, &args, &arg_ranges);
        }

        let cross_product_iterator = MultisetCrossProductIterator::new(
            pools.iter().map(|p| p.multiset_iterator()).collect(),
        );

        let mut args = args.clone();
        let mut results = Vec::new();
        for (values, weight) in cross_product_iterator {
            for ((i, is_element), value) in pool_iterator_info.iter().zip(values.iter()) {
                if *is_element {
                    args[*i] = RuntimeValue::Element(value[0].clone());
                } else {
                    args[*i] = RuntimeValue::List(Rc::new(reverse_if(!self.lowest_first, value)));
                }
            }
            results.push((
                self.call_function(eval_context, function, &args, &arg_ranges)?,
                weight,
            ));
        }

        // Each result becomes one distribution, and the distributions are
        // mixed. Nothing requires the results to agree with each other: a call
        // that returns a number for one multiset and a tuple for another
        // produces a die that holds both, and it is displaying or summing that
        // die which says they cannot be reconciled.
        const COMBINING: &str = "combining the results of a function evaluated over a pool";
        let mut result_distributions = Vec::new();
        for (result, weight) in results {
            if let RuntimeValue::Pool(ref pool) = result {
                // The empty die is ignored in this context, but the empty list is not.
                if pool.ordered_outcomes().is_empty() {
                    continue;
                }
            }
            let pool = result.to_pool().map_err(|mismatch| {
                mismatch.into_error(
                    function.range,
                    COMBINING,
                    NonAdditiveSubject::Sequence(result.elements().len()),
                )
            })?;
            let summed =
                sum_pool(&pool).map_err(|error| error.into_error(function.range, COMBINING))?;
            result_distributions.push((weight, summed));
        }
        Ok(RuntimeValue::Pool(Rc::new(Pool::from_mixture(
            result_distributions,
        ))))
    }

    fn call_function(
        &mut self,
        eval_context: &EvalContext,
        function: &WithRange<Function>,
        args: &[RuntimeValue],
        arg_ranges: &[ast::Range],
    ) -> Result<RuntimeValue, RuntimeError> {
        match &function.value {
            Function::Primitive(primitive) => {
                if transform_depth_affects_result(primitive.identifier, args) {
                    self.warn_limit(
                        eval_context.source_id,
                        function.range,
                        WarningKind::ExplodeDepth,
                        "explode depth",
                        self.explode_depth,
                        "Explode depth bounds this distribution",
                    );
                }
                (primitive.execute)(
                    args,
                    crate::primitives::PrimitiveCtx {
                        arg_ranges,
                        explode_depth: self.explode_depth,
                        lowest_first: self.lowest_first,
                        function_range: function.range,
                        identifier: primitive.identifier,
                    },
                )
            }
            Function::UserDefined(user_function) => {
                self.env.push_frame();
                for (arg, formal) in args.iter().zip(user_function.definition.args.iter()) {
                    self.env.insert(formal.value.name.clone(), arg.clone());
                }
                let new_context = EvalContext {
                    recursion_depth: eval_context.recursion_depth + 1,
                    block_depth: eval_context.block_depth + 1,
                    source_id: user_function.source_id,
                };
                let result: Result<RuntimeValue, RuntimeError> = (|| {
                    let mut result = None;
                    for statement in &user_function.definition.body {
                        result = self.execute_statement(&new_context, statement)?;
                        if result.is_some() {
                            break;
                        }
                    }
                    // If there's no result, there was no return statement in the function.
                    Ok(result.unwrap_or_else(RuntimeValue::empty_pool))
                })();
                self.env.pop_frame();
                result.map_err(|source| {
                    let range = source.range();
                    RuntimeError::InFunction {
                        range: range.into(),
                        source: Box::new(source),
                        body_source: user_function.source_id,
                        frame: Box::new(EvaluationFrame {
                            function: user_function.definition.name.value.clone(),
                            call: SourceRange {
                                source: eval_context.source_id,
                                range: function.range,
                            },
                            definition: Some(SourceRange {
                                source: user_function.source_id,
                                range: user_function.definition.name.range,
                            }),
                            bindings: user_function
                                .definition
                                .args
                                .iter()
                                .zip(args)
                                .map(|(formal, value)| TraceBinding {
                                    name: formal.value.name.clone(),
                                    value: preview_value(value, &self.symbols),
                                })
                                .collect(),
                        }),
                    }
                })
            }
        }
    }
}

/// Splits a function identifier such as `"field {} of {}"` into its words and
/// its number of argument positions.
fn identifier_shape(identifier: &str) -> (String, usize) {
    let mut words = Vec::new();
    let mut arity = 0;
    for token in identifier.split_whitespace() {
        if token == "{}" {
            arity += 1;
        } else {
            words.push(token);
        }
    }
    (words.join(" "), arity)
}

/// Detects a call that did not resolve, but whose words name a function taking
/// a different number of arguments.
///
/// This is overwhelmingly caused by two adjacent arguments being parsed as one
/// expression: `d` binds tighter than argument separation, so `[tuple d6 d8]`
/// reads as the single argument `(d6)d8`. Reporting the mismatch turns a
/// confusing "no such function" into an actionable fix.
fn arity_mismatch(
    functions: &HashMap<String, Function>,
    name: &str,
    args: &[WithRange<Expression>],
) -> Option<ArityMismatch> {
    let (words, arity) = identifier_shape(name);
    let mut available = functions
        .keys()
        .filter_map(|identifier| {
            let (other_words, other_arity) = identifier_shape(identifier);
            (other_words == words && other_arity != arity).then_some(other_arity)
        })
        .collect::<Vec<_>>();
    if available.is_empty() {
        return None;
    }
    available.sort_unstable();
    available.dedup();
    // A comma can only be missing if the call passed too few arguments; if
    // every known arity is smaller, the caller passed too many instead.
    let comma_insertion = available
        .iter()
        .any(|other| *other > arity)
        .then(|| comma_insertion(args))
        .flatten();
    Some(ArityMismatch {
        words,
        available,
        found: arity,
        comma_insertion,
    })
}

/// Where a comma would split a single argument that is really two dice.
///
/// `d` binds tighter than argument separation, so `[tuple d6 d8]` parses as the
/// one argument `(d6)d8`. Wherever the left side of a `d` is itself a dice
/// roll, the two sides were written as separate arguments, and the leftmost
/// such `d` is where the first comma belongs. Reading this off the parse tree
/// rather than the source text means the fix cannot disagree with how the call
/// was actually parsed.
fn comma_insertion(args: &[WithRange<Expression>]) -> Option<usize> {
    let [only] = args else {
        return None;
    };
    let mut expression = &only.value;
    let mut insertion = None;
    while let Expression::BinaryOp { op, left, .. } = expression {
        if op.value != ast::BinaryOp::D || !rolls_dice(&left.value) {
            break;
        }
        insertion = Some(left.range.end);
        expression = &left.value;
    }
    insertion
}

/// Whether an expression rolls dice at its outermost level, so that a `d` to
/// its right cannot have been reading it as a die count.
fn rolls_dice(expression: &Expression) -> bool {
    match expression {
        Expression::UnaryOp { op, .. } => op.value == ast::UnaryOp::D,
        Expression::BinaryOp { op, left, .. } => {
            op.value == ast::BinaryOp::D || rolls_dice(&left.value)
        }
        _ => false,
    }
}

/// Rejects an output whose outcomes cannot share one schema.
///
/// A displayed distribution has one shape and one schema per field position, so
/// this is the last chance to reject outcomes that do not agree on either — and
/// the only one, since nothing on the way here required them to. Two things are
/// rejected:
///
/// - outcomes of different *shapes*, a number in one and a tuple in another, or
///   tuples of different arity. `{1, [tuple 1 2]}` is a perfectly good sequence
///   that `print` will show; it just cannot be drawn as one distribution.
/// - a field that is a number in one outcome and a symbol in another, which has
///   no single way to be labelled. A field that is *all* symbols is an ordinary
///   categorical output and is fine.
///
/// [`crate::output`] runs after evaluation and cannot report an error, so
/// anything it would rather not meet has to be stopped here.
fn reject_mixed_output_fields(
    outcomes: &[&ElementValue],
    range: ast::Range,
) -> Result<(), RuntimeError> {
    let shape_of = |outcome: &ElementValue| match outcome {
        ElementValue::Tuple(fields) => Some(fields.len()),
        _ => None,
    };
    // The empty sum has no shape of its own — it takes the one around it — so
    // it is not what settles the shape and never disagrees with it.
    let shaped = |outcome: &&&ElementValue| !matches!(outcome, ElementValue::AdditiveIdentity);
    let Some(first) = outcomes.iter().find(shaped) else {
        return Ok(());
    };
    let shape = shape_of(first);
    if let Some(other) = outcomes
        .iter()
        .filter(shaped)
        .find(|outcome| shape_of(outcome) != shape)
    {
        return Err(RuntimeError::ShapeMismatch(Box::new(ShapeMismatchError {
            range: range.into(),
            action: "displaying a distribution",
            first: (*first).clone(),
            second: (*other).clone(),
        })));
    }
    let arity = shape.unwrap_or(1);
    let mut seen = vec![(false, false); arity];
    for outcome in outcomes {
        let fields: &[ElementValue] = match outcome {
            ElementValue::Tuple(fields) => fields,
            other => std::slice::from_ref(other),
        };
        for (position, field) in fields.iter().enumerate() {
            let (numbers, symbols) = &mut seen[position];
            match field {
                ElementValue::Symbol(_) => *symbols = true,
                _ => *numbers = true,
            }
        }
    }
    let Some(position) = seen
        .iter()
        .position(|(numbers, symbols)| *numbers && *symbols)
    else {
        return Ok(());
    };
    let what = if arity == 1 {
        "this distribution".to_string()
    } else {
        format!("field {} of this distribution", position + 1)
    };
    Err(RuntimeError::Semantic {
        kind: SemanticErrorKind::OutcomeMismatch,
        range: range.into(),
        message: format!(
            "{what} holds numbers in some outcomes and symbols in others, so it cannot be \
             displayed; map it to one kind of value first, for instance with \
             `[sum integers in ...]`, or use `print` to inspect it as it is"
        ),
    })
}

/// Where a list element was written, so a diagnostic can point at it.
fn list_item_range(item: &ListItem) -> ast::Range {
    match &item.item {
        BareListItem::Expr(expr) => expr.range,
        BareListItem::Range(start, end) => ast::Range {
            start: start.range.start,
            end: end.range.end,
        },
    }
}

/// Converts an argument to the shape its parameter is declared with.
///
/// Only the shape: `:n`, `:s` and `:d` say how a value arrives, never what it
/// holds. The one conversion that can fail is summing, which a sequence or pool
/// of values that cannot be added discovers here.
fn coerce_arg(
    arg: RuntimeValue,
    expected: Option<StaticType>,
    range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    let Some(expected) = expected else {
        return Ok(arg);
    };
    const SUMMING: &str = "passing a value to a parameter that expects a single value";
    let coerced = match (arg, expected) {
        (value @ RuntimeValue::Element(_), StaticType::Int) => Ok(value),
        (RuntimeValue::Element(value), StaticType::List) => {
            Ok(RuntimeValue::List(Rc::new(vec![value])))
        }
        (RuntimeValue::Element(value), StaticType::Pool) => {
            Ok(RuntimeValue::Pool(Rc::new(Pool::from_list(1, vec![value]))))
        }
        (RuntimeValue::List(list), StaticType::Int) => sum_elements(&list)
            .map(RuntimeValue::Element)
            .map_err(|mismatch| {
                mismatch.into_error(range, SUMMING, NonAdditiveSubject::Sequence(list.len()))
            }),
        (value @ RuntimeValue::List(_), StaticType::List) => Ok(value),
        (RuntimeValue::List(list), StaticType::Pool) => Ok(RuntimeValue::Pool(Rc::new(
            Pool::from_list(1, (*list).clone()),
        ))),
        (RuntimeValue::Pool(pool), StaticType::Int) => {
            let summed = sum_pool(&pool).map_err(|error| error.into_error(range, SUMMING))?;
            Ok(RuntimeValue::Pool(Rc::new(summed)))
        }
        (value @ RuntimeValue::Pool(_), _) => Ok(value),
    }?;
    // Where a single value is required and the empty sum has still met nothing
    // to take its shape from, it becomes the integer 0.
    if expected == StaticType::Int
        && matches!(
            coerced,
            RuntimeValue::Element(ElementValue::AdditiveIdentity)
        )
    {
        return Ok(RuntimeValue::Element(ElementValue::Int(0)));
    }
    Ok(coerced)
}

fn interpolate_variable_names(
    template: &WithRange<String>,
    vars: &ValEnv,
    symbols: &SymbolTable,
) -> Result<String, RuntimeError> {
    let mut result = String::new();
    let mut char_iter = template.value.chars();
    while let Some(chr) = char_iter.next() {
        if chr == '[' {
            let mut valid_end = false;
            let mut name = String::new();
            for chr in char_iter.by_ref() {
                if chr == ']' {
                    valid_end = true;
                    break;
                }
                name.push(chr);
            }
            if !valid_end
                || name.is_empty()
                || name.chars().any(|c| !c.is_ascii_uppercase() && c != '_')
            {
                result.push('[');
                result.push_str(&name);
                if valid_end {
                    result.push(']');
                }
            } else if let Some(value) = vars.get(&name) {
                // A name is prose, so the empty sum reads better as the `0` it
                // stands for than as `𝑒`.
                let value = match value {
                    RuntimeValue::Element(ElementValue::AdditiveIdentity) => {
                        RuntimeValue::Element(ElementValue::Int(0))
                    }
                    value => value,
                };
                result.push_str(&value.display(symbols).to_string());
            } else {
                return Err(RuntimeError::UndefinedReference {
                    range: template.range.into(),
                    name,
                });
            }
        } else {
            result.push(chr);
        }
    }
    Ok(result)
}

fn reverse_if<T: Clone>(should_reverse: bool, values: &[T]) -> Vec<T> {
    if should_reverse {
        values.iter().cloned().rev().collect()
    } else {
        values.into()
    }
}

fn same_pool(left: &RuntimeValue, right: &RuntimeValue) -> bool {
    matches!(
        (left, right),
        (RuntimeValue::Pool(left), RuntimeValue::Pool(right))
            if Rc::ptr_eq(left, right)
    )
}

fn transform_depth_affects_result(identifier: &str, args: &[RuntimeValue]) -> bool {
    // Taken from the registry rather than matched by prefix, so renaming or
    // adding a primitive cannot silently change which calls are bounded.
    let bounded_by_depth = [
        EXPLODE_PRIMITIVE.identifier,
        EXPLODE_ON_PRIMITIVE.identifier,
        REROLL_PRIMITIVE.identifier,
        REROLL_ON_PRIMITIVE.identifier,
    ];
    if !bounded_by_depth.contains(&identifier) {
        return false;
    }
    let Some(RuntimeValue::Pool(pool)) = args.first() else {
        return false;
    };
    if pool.ordered_outcomes().is_empty() {
        return false;
    }
    let Some(condition) = args.get(1) else {
        // Bare explode/reroll triggers on the highest outcome, which exists for
        // every nonempty pool.
        return true;
    };
    let RuntimeValue::List(condition) = condition else {
        return false;
    };
    pool.ordered_outcomes()
        .iter()
        .any(|(outcome, _)| condition.contains(outcome))
}

fn closest_names<'a>(needle: &str, candidates: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut ranked = candidates
        .map(|candidate| (edit_distance(needle, candidate), candidate))
        .collect::<Vec<_>>();
    ranked.sort();
    ranked.dedup_by(|left, right| left.1 == right.1);
    let threshold = 2.max(needle.chars().count() / 3);
    ranked
        .into_iter()
        .filter(|(distance, _)| *distance <= threshold)
        .take(3)
        .map(|(_, candidate)| candidate.to_string())
        .collect()
}

fn edit_distance(left: &str, right: &str) -> usize {
    let right = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_character) in left.chars().enumerate() {
        let mut current = vec![left_index + 1];
        for (right_index, right_character) in right.iter().enumerate() {
            current.push(
                (previous[right_index + 1] + 1)
                    .min(current[right_index] + 1)
                    .min(previous[right_index] + usize::from(left_character != *right_character)),
            );
        }
        previous = current;
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_variable_names() {
        fn ranged(s: &str) -> WithRange<String> {
            WithRange {
                value: s.to_string(),
                range: ast::Range {
                    start: 0,
                    end: s.len(),
                },
            }
        }

        let symbols = SymbolTable::default();
        let mut env = ValEnv::new();
        env.insert("A".to_string(), 1.into());
        env.insert("B".to_string(), 2.into());
        env.insert("C".to_string(), 3.into());
        env.insert("_MY_VAR".to_string(), 4.into());
        env.insert(
            "IDENTITY".to_string(),
            RuntimeValue::Element(ElementValue::AdditiveIdentity),
        );
        assert_eq!(
            interpolate_variable_names(&ranged("A + B = [A] + [B]"), &env, &symbols).unwrap(),
            "A + B = 1 + 2"
        );
        assert!(
            interpolate_variable_names(&ranged("A + B = [A] + [B] + [INVALID]"), &env, &symbols)
                .is_err()
        );
        assert_eq!(
            interpolate_variable_names(&ranged("A + B = [A] + [B] + [invalid]"), &env, &symbols)
                .unwrap(),
            "A + B = 1 + 2 + [invalid]"
        );
        assert_eq!(
            interpolate_variable_names(&ranged("A + B = [A] + [B] + [C"), &env, &symbols).unwrap(),
            "A + B = 1 + 2 + [C"
        );
        assert_eq!(
            interpolate_variable_names(&ranged("[_MY_VAR], [IDENTITY]"), &env, &symbols).unwrap(),
            "4, 0"
        );
    }

    #[test]
    fn additive_identity_becomes_zero_where_an_argument_must_be_an_int() {
        let range = ast::Range { start: 0, end: 0 };

        for argument in [
            RuntimeValue::Element(ElementValue::AdditiveIdentity),
            // An empty sequence sums to the empty sum on its way to an int.
            RuntimeValue::empty_list(),
        ] {
            assert_eq!(
                coerce_arg(argument.clone(), Some(StaticType::Int), range).unwrap(),
                RuntimeValue::Element(ElementValue::Int(0)),
                "{:?}",
                argument
            );
        }
    }

    #[test]
    fn additive_identity_is_inspectable_in_print_but_defaulted_in_output() {
        use std::{cell::RefCell, rc::Rc};

        let printed = Rc::new(RefCell::new(None));
        let callback_result = Rc::clone(&printed);
        let mut evaluator = Evaluator::new();
        evaluator.set_print_callback(Box::new(move |value, name| {
            *callback_result.borrow_mut() = Some((value.to_string(), name));
        }));

        let statements = crate::grammar::BodyParser::new()
            .parse(r#"X: {} + {} print X named "[X]" output X named "[X]""#)
            .unwrap();
        for statement in statements {
            evaluator.execute(&statement).unwrap();
        }

        assert_eq!(
            printed.borrow().as_ref().unwrap(),
            &("\u{1d452}".to_string(), "0".to_string())
        );
        let output = evaluator.take_outputs().remove(0);
        assert_eq!(output.name, "0");
        assert_eq!(
            crate::output::Distribution::from_runtime(output.value, None, evaluator.symbols())
                .probabilities,
            [(vec![0], 1.0)]
        );
    }

    #[test]
    fn test_division_by_zero() {
        use crate::ast::{BinaryOp, Range};

        let left = 5.into();
        let right = 0.into();
        let op = WithRange {
            value: BinaryOp::Div,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Cannot divide 5 by zero"));
            }
            _ => panic!("Expected MathError for division by zero"),
        }
    }

    #[test]
    fn test_negative_exponentiation() {
        use crate::ast::{BinaryOp, Range};

        let left = 2.into();
        let right = (-3).into();
        let op = WithRange {
            value: BinaryOp::Pow,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Cannot raise 2 to negative power -3"));
            }
            _ => panic!("Expected MathError for negative exponentiation"),
        }
    }

    /// Every overflowing operation names itself and the operands that
    /// overflowed, so the message points at the arithmetic that failed.
    #[test]
    fn test_arithmetic_overflow() {
        use crate::ast::{BinaryOp, Range};

        for (binary_op, left, right, expected_operation, expected_operands) in [
            (
                BinaryOp::Add,
                i32::MAX,
                1,
                "Addition overflow",
                i32::MAX.to_string(),
            ),
            (
                BinaryOp::Sub,
                i32::MIN,
                1,
                "Subtraction overflow",
                i32::MIN.to_string(),
            ),
            (
                BinaryOp::Mul,
                i32::MAX,
                2,
                "Multiplication overflow",
                i32::MAX.to_string(),
            ),
            // 2^32 overflows i32.
            (BinaryOp::Pow, 2, 32, "Power overflow", "2 ^ 32".to_string()),
        ] {
            let range = Range { start: 0, end: 1 };
            let op = WithRange {
                value: binary_op,
                range,
            };
            let left: RuntimeValue = left.into();
            let right: RuntimeValue = right.into();

            let result = apply_binary_op(&op, &left, range, &right, false);

            let Err(RuntimeError::MathError { message, .. }) = result else {
                panic!("expected a MathError for {expected_operation}, got {result:?}");
            };
            assert!(message.contains(expected_operation), "{message}");
            assert!(message.contains(&expected_operands), "{message}");
        }
    }
}
