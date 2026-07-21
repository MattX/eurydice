//! A simple tree walking interpreter.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
};

use malachite::base::num::arithmetic::traits::Lcm;
use malachite::{
    base::num::basic::traits::{One, Zero},
    rational::Rational,
    Natural,
};
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{
        self, BareListItem, BinaryOp, EnumDefinition, Expression, FunctionDefinition, ListItem,
        PositionOrder, SetParam, Statement, StaticType, UnaryOp, WithRange,
    },
    dice::{MultisetCrossProductIterator, Pool},
    primitives::{register_primitives, Primitive},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EnumType {
    pub name: String,
    pub members: Vec<String>,
}

impl EnumType {
    pub fn member_name(&self, value: i32) -> Option<&str> {
        usize::try_from(value)
            .ok()
            .and_then(|index| self.members.get(index))
            .map(String::as_str)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarType {
    Int,
    Enum(Rc<EnumType>),
    Tuple(Rc<[ScalarType]>),
}

impl ScalarType {
    fn merged_with(&self, other: &Self) -> Option<Self> {
        if self == other {
            Some(self.clone())
        } else {
            None
        }
    }

    fn display_name(&self) -> String {
        match self {
            ScalarType::Int => "int".to_string(),
            ScalarType::Enum(ty) => ty.name.clone(),
            ScalarType::Tuple(fields) => format!(
                "tuple({})",
                fields
                    .iter()
                    .map(ScalarType::display_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarValue {
    Int(i32),
    Enum { value: i32, ty: Rc<EnumType> },
    Tuple(Rc<[ScalarValue]>),
}

impl ScalarValue {
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            ScalarValue::Int(_) => ScalarType::Int,
            ScalarValue::Enum { ty, .. } => ScalarType::Enum(Rc::clone(ty)),
            ScalarValue::Tuple(fields) => ScalarType::Tuple(
                fields
                    .iter()
                    .map(ScalarValue::scalar_type)
                    .collect::<Vec<_>>()
                    .into(),
            ),
        }
    }

    pub fn as_int(&self) -> Option<i32> {
        match self {
            ScalarValue::Int(value) => Some(*value),
            ScalarValue::Enum { .. } | ScalarValue::Tuple(_) => None,
        }
    }

    pub fn enum_type(&self) -> Option<Rc<EnumType>> {
        match self {
            ScalarValue::Enum { ty, .. } => Some(Rc::clone(ty)),
            ScalarValue::Int(_) | ScalarValue::Tuple(_) => None,
        }
    }
}

impl std::fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarValue::Int(value) => write!(f, "{value}"),
            ScalarValue::Enum { value, ty } => {
                write!(f, "{}", ty.member_name(*value).unwrap_or("<?>"))
            }
            ScalarValue::Tuple(fields) => write!(
                f,
                "({})",
                fields
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    Scalar(ScalarValue),
    List(Rc<Vec<ScalarValue>>, ScalarType),
    Pool(Rc<Pool<ScalarValue>>, ScalarType),
}

impl std::fmt::Display for RuntimeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeValue::Scalar(value) => write!(f, "{value}"),
            RuntimeValue::List(list, _) => write!(
                f,
                "{{{}}}",
                list.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            RuntimeValue::Pool(pool, ScalarType::Int) => write!(
                f,
                "{}",
                (**pool)
                    .clone()
                    .map_outcomes(|outcome| outcome.as_int().expect("numeric pool"))
            ),
            RuntimeValue::Pool(pool, ScalarType::Enum(_) | ScalarType::Tuple(_)) => write!(
                f,
                "d{{{}}}",
                pool.ordered_outcomes()
                    .iter()
                    .map(|(outcome, count)| format!("{outcome}:{count}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl RuntimeValue {
    fn runtime_type(&self) -> StaticType {
        match self {
            RuntimeValue::Scalar(_) => StaticType::Int,
            RuntimeValue::List(_, _) => StaticType::List,
            RuntimeValue::Pool(_, _) => StaticType::Pool,
        }
    }

    fn outcome_type(&self) -> ScalarType {
        match self {
            RuntimeValue::Scalar(value) => value.scalar_type(),
            RuntimeValue::List(_, outcome_type) | RuntimeValue::Pool(_, outcome_type) => {
                outcome_type.clone()
            }
        }
    }

    pub fn enum_type(&self) -> Option<Rc<EnumType>> {
        match self.outcome_type() {
            ScalarType::Enum(ty) => Some(ty),
            ScalarType::Int | ScalarType::Tuple(_) => None,
        }
    }

    pub(crate) fn has_same_outcome_type(&self, other: &Self) -> bool {
        self.outcome_type() == other.outcome_type()
    }

    fn is_numeric(&self) -> bool {
        self.outcome_type() == ScalarType::Int
    }

    fn to_list(&self, repeat: usize) -> Vec<ScalarValue> {
        match self {
            RuntimeValue::Scalar(value) => vec![value.clone(); repeat],
            RuntimeValue::List(list, _) => (0..repeat).flat_map(|_| list.iter().cloned()).collect(),
            RuntimeValue::Pool(pool, _) => {
                let outcomes = sum_pool(pool)
                    .into_die_iter()
                    .map(|(outcome, _)| outcome)
                    .collect::<Vec<_>>();
                (0..repeat).flat_map(|_| outcomes.iter().cloned()).collect()
            }
        }
    }

    pub(crate) fn map_numeric_outcomes(&self, f: impl Fn(i32) -> i32 + Copy) -> Self {
        match self {
            RuntimeValue::Scalar(ScalarValue::Int(value)) => f(*value).into(),
            RuntimeValue::List(list, _) => f(list.iter().map(expect_int).sum()).into(),
            RuntimeValue::Pool(pool, _) => RuntimeValue::Pool(
                Rc::new(
                    (**pool)
                        .clone()
                        .map_outcomes(|outcome| ScalarValue::Int(f(expect_int(&outcome)))),
                ),
                ScalarType::Int,
            ),
            RuntimeValue::Scalar(ScalarValue::Enum { .. } | ScalarValue::Tuple(_)) => {
                unreachable!("non-numeric values are rejected before numeric mapping")
            }
        }
    }

    fn to_pool(&self) -> Pool<ScalarValue> {
        match self {
            RuntimeValue::Scalar(value) => Pool::from_list(1, vec![value.clone()]),
            RuntimeValue::List(list, _) if self.is_numeric() => {
                Pool::from_list(1, vec![ScalarValue::Int(list.iter().map(expect_int).sum())])
            }
            RuntimeValue::List(list, _) => Pool::from_list(1, (**list).clone()),
            RuntimeValue::Pool(pool, _) => (**pool).clone(),
        }
    }
}

fn expect_int(value: &ScalarValue) -> i32 {
    value
        .as_int()
        .expect("numeric operation received a non-numeric scalar")
}

pub(crate) fn sum_pool(pool: &Pool<ScalarValue>) -> Pool<ScalarValue> {
    if pool.is_empty() {
        return Pool::from_list(1, vec![ScalarValue::Int(0)]);
    }
    if pool.dimension() == 1 {
        return pool.clone();
    }
    pool.clone()
        .map_outcomes(|outcome| expect_int(&outcome))
        .sum()
        .map_outcomes(ScalarValue::Int)
}

impl From<i32> for RuntimeValue {
    fn from(value: i32) -> Self {
        RuntimeValue::Scalar(ScalarValue::Int(value))
    }
}

impl From<Rc<Vec<i32>>> for RuntimeValue {
    fn from(value: Rc<Vec<i32>>) -> Self {
        RuntimeValue::List(
            Rc::new(value.iter().copied().map(ScalarValue::Int).collect()),
            ScalarType::Int,
        )
    }
}

impl From<Vec<i32>> for RuntimeValue {
    fn from(value: Vec<i32>) -> Self {
        RuntimeValue::List(
            Rc::new(value.into_iter().map(ScalarValue::Int).collect()),
            ScalarType::Int,
        )
    }
}

impl From<Pool> for RuntimeValue {
    fn from(value: Pool) -> Self {
        RuntimeValue::Pool(
            Rc::new(value.map_outcomes(ScalarValue::Int)),
            ScalarType::Int,
        )
    }
}

/// Represents the environment at some point during program execution.
///
/// Lambdas capture their arguments, so we need RCs to keep the environments alive;
/// we also need RefCells because the bindings are mutable.
#[derive(Debug)]
struct ValEnv {
    parent: Option<RcValEnv>,
    env: HashMap<String, RuntimeValue>,
}
type RcValEnv = Rc<RefCell<ValEnv>>;

impl ValEnv {
    fn new() -> Self {
        Self {
            parent: None,
            env: HashMap::new(),
        }
    }

    fn with_parent(parent: RcValEnv) -> Self {
        Self {
            parent: Some(parent),
            env: HashMap::new(),
        }
    }

    fn get(&self, key: &str) -> Option<RuntimeValue> {
        if let Some(value) = self.env.get(key) {
            Some(value.clone())
        } else if let Some(parent) = &self.parent {
            parent.borrow().get(key)
        } else {
            None
        }
    }

    fn insert(&mut self, key: String, value: RuntimeValue) {
        self.env.insert(key, value);
    }

    fn contains(&self, key: &str) -> bool {
        self.env.contains_key(key)
            || self
                .parent
                .as_ref()
                .is_some_and(|parent| parent.borrow().contains(key))
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
    arg_types: Vec<Option<ResolvedArgType>>,
}

#[derive(Debug, Clone)]
struct ResolvedArgType {
    shape: StaticType,
    outcome: Option<ScalarType>,
}

impl Function {
    fn get_arg_types(&self) -> Vec<Option<ResolvedArgType>> {
        match self {
            Function::Primitive(primitive) => primitive
                .arg_types
                .iter()
                .map(|shape| {
                    shape.map(|shape| ResolvedArgType {
                        shape,
                        outcome: (!primitive.accepts_non_numeric).then_some(ScalarType::Int),
                    })
                })
                .collect(),
            Function::UserDefined(function) => function.arg_types.clone(),
        }
    }
}

struct EvalContext {
    env: RcValEnv,
    recursion_depth: usize,
    block_depth: usize,
}

impl EvalContext {
    fn new(env: RcValEnv) -> Self {
        Self {
            env,
            recursion_depth: 0,
            block_depth: 0,
        }
    }

    fn nested_block(&self) -> Self {
        Self {
            env: Rc::clone(&self.env),
            recursion_depth: self.recursion_depth,
            block_depth: self.block_depth + 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BindingKind {
    Assignment,
    FunctionParameter,
    LoopVariable,
}

#[derive(Debug, Clone, Copy)]
enum EnumIdentifierKind {
    Type,
    Member,
}

impl std::fmt::Display for EnumIdentifierKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnumIdentifierKind::Type => write!(f, "type"),
            EnumIdentifierKind::Member => write!(f, "member"),
        }
    }
}

pub struct Evaluator {
    global_env: RcValEnv,
    outputs: Vec<(RuntimeValue, String)>,
    functions: HashMap<String, Function>,
    enums: HashMap<String, Rc<EnumType>>,
    enum_members: HashSet<String>,
    explode_depth: usize,
    recursion_depth: usize,
    lowest_first: bool,
    print_callback: Option<Box<dyn Fn(RuntimeValue, String)>>,
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
            global_env: Rc::new(RefCell::new(ValEnv::new())),
            outputs: Vec::new(),
            functions,
            enums: HashMap::new(),
            enum_members: HashSet::new(),
            explode_depth: 2,
            recursion_depth: 10,
            lowest_first: false,
            print_callback: None,
        }
    }

    pub fn set_print_callback(&mut self, callback: Box<dyn Fn(RuntimeValue, String)>) {
        self.print_callback = Some(callback);
    }

    pub fn execute(&mut self, statement: &WithRange<Statement>) -> Result<(), RuntimeError> {
        let eval_context = EvalContext::new(Rc::clone(&self.global_env));
        let result = self.execute_statement(&eval_context, statement)?;
        debug_assert!(
            result.is_none(),
            "value returned from a top-level statement"
        );
        Ok(())
    }

    pub fn take_outputs(&mut self) -> Vec<(RuntimeValue, String)> {
        std::mem::take(&mut self.outputs)
    }

    fn enum_identifier_kind(&self, name: &str) -> Option<EnumIdentifierKind> {
        if self.enums.contains_key(name) {
            Some(EnumIdentifierKind::Type)
        } else if self.enum_members.contains(name) {
            Some(EnumIdentifierKind::Member)
        } else {
            None
        }
    }

    fn validate_binding_name(
        &self,
        name: &str,
        range: ast::Range,
        binding: BindingKind,
    ) -> Result<(), RuntimeError> {
        let Some(identifier_kind) = self.enum_identifier_kind(name) else {
            return Ok(());
        };
        let message = match binding {
            BindingKind::Assignment => {
                format!("enum {identifier_kind} {name} is immutable")
            }
            BindingKind::FunctionParameter => {
                format!("enum {identifier_kind} {name} cannot be used as a parameter")
            }
            BindingKind::LoopVariable => {
                format!("enum {identifier_kind} {name} cannot be used as a loop variable")
            }
        };
        Err(RuntimeError::EnumTypeError {
            range: range.into(),
            message,
        })
    }

    fn define_enum(
        &mut self,
        eval_context: &EvalContext,
        definition: &EnumDefinition,
        statement_range: ast::Range,
    ) -> Result<(), RuntimeError> {
        if eval_context.recursion_depth != 0 || eval_context.block_depth != 0 {
            return Err(RuntimeError::EnumTypeError {
                range: statement_range.into(),
                message: "enum declarations are only allowed at the top level".to_string(),
            });
        }
        if let Some(kind) = self.enum_identifier_kind(&definition.name.value) {
            return Err(RuntimeError::EnumTypeError {
                range: definition.name.range.into(),
                message: format!(
                    "enum {} conflicts with an existing enum {kind}",
                    definition.name.value
                ),
            });
        }
        if eval_context.env.borrow().contains(&definition.name.value) {
            return Err(RuntimeError::EnumTypeError {
                range: definition.name.range.into(),
                message: format!("{} is already bound as a variable", definition.name.value),
            });
        }

        let mut seen = HashSet::new();
        for member in &definition.members {
            if !seen.insert(member.value.as_str()) {
                return Err(RuntimeError::EnumTypeError {
                    range: member.range.into(),
                    message: format!("enum member {} is already defined", member.value),
                });
            }
            if member.value == definition.name.value {
                return Err(RuntimeError::EnumTypeError {
                    range: member.range.into(),
                    message: format!("enum member {} conflicts with its enum type", member.value),
                });
            }
            if let Some(kind) = self.enum_identifier_kind(&member.value) {
                return Err(RuntimeError::EnumTypeError {
                    range: member.range.into(),
                    message: format!(
                        "enum member {} conflicts with an existing enum {kind}",
                        member.value
                    ),
                });
            }
            if eval_context.env.borrow().contains(&member.value) {
                return Err(RuntimeError::EnumTypeError {
                    range: member.range.into(),
                    message: format!("{} is already bound as a variable", member.value),
                });
            }
        }

        let ty = Rc::new(EnumType {
            name: definition.name.value.clone(),
            members: definition.members.iter().map(|m| m.value.clone()).collect(),
        });
        for (index, member) in definition.members.iter().enumerate() {
            let value = i32::try_from(index).map_err(|_| RuntimeError::EnumTypeError {
                range: member.range.into(),
                message: "enum has too many members".to_string(),
            })?;
            self.global_env.borrow_mut().insert(
                member.value.clone(),
                RuntimeValue::Scalar(ScalarValue::Enum {
                    value,
                    ty: Rc::clone(&ty),
                }),
            );
            self.enum_members.insert(member.value.clone());
        }
        self.enums.insert(definition.name.value.clone(), ty);
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
                self.validate_binding_name(&name.value, name.range, BindingKind::Assignment)?;
                let value = self.evaluate(eval_context, value)?;
                eval_context
                    .env
                    .borrow_mut()
                    .insert(name.value.clone(), value);
            }
            Statement::FunctionDefinition(fd) => {
                for arg in &fd.args {
                    self.validate_binding_name(
                        &arg.value.name,
                        arg.range,
                        BindingKind::FunctionParameter,
                    )?;
                }
                let arg_types = fd
                    .args
                    .iter()
                    .map(|arg| {
                        arg.value.ty.map(|shape| ResolvedArgType {
                            shape,
                            outcome: None,
                        })
                    })
                    .collect();
                self.functions.insert(
                    fd.name.value.clone(),
                    Function::UserDefined(Rc::new(UserFunction {
                        definition: fd.clone(),
                        arg_types,
                    })),
                );
            }
            Statement::EnumDefinition(definition) => {
                self.define_enum(eval_context, definition, statement.range)?;
            }
            Statement::Output { expr, named } => {
                if eval_context.recursion_depth != 0 {
                    return Err(RuntimeError::OutputNotAtTopLevel {
                        range: statement.range.into(),
                    });
                }
                let value = self.evaluate(eval_context, expr)?;
                let name = match named {
                    Some(name) => interpolate_variable_names(name, &eval_context.env)?,
                    None => format!("output {}", self.outputs.len() + 1),
                };
                self.outputs.push((value, name));
            }
            Statement::Print { expr, named } => {
                let value = self.evaluate(eval_context, expr)?;
                let name = match named {
                    Some(name) => interpolate_variable_names(name, &eval_context.env)?,
                    None => "".to_string(),
                };
                if let Some(ref callback) = self.print_callback {
                    callback(value, name);
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
                    RuntimeValue::Scalar(ScalarValue::Int(i)) => i,
                    _ => {
                        return Err(RuntimeError::InvalidCondition {
                            range: condition.range.into(),
                            found: condition_value.runtime_type(),
                            value: condition_value,
                        })
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
                self.validate_binding_name(
                    &variable.value,
                    variable.range,
                    BindingKind::LoopVariable,
                )?;
                let range = self.evaluate(eval_context, range_expression)?;
                let range = match range {
                    RuntimeValue::List(range, _) => range,
                    _ => {
                        return Err(RuntimeError::LoopOverNonSequence {
                            range: range_expression.range.into(),
                            found: range.runtime_type(),
                        })
                    }
                };
                let nested_context = eval_context.nested_block();
                for value in range.iter() {
                    eval_context
                        .env
                        .borrow_mut()
                        .insert(variable.value.clone(), RuntimeValue::Scalar(value.clone()));
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
                let mut outcome: Option<ScalarType> = None;
                let mut elems = Vec::new();
                for item in items {
                    let item_type = item.outcome_type();
                    merge_outcome_type(
                        &mut outcome,
                        item_type,
                        expression.range,
                        "list literal contains mixed outcome types",
                    )?;
                    elems.extend(item.to_list(1));
                }
                Ok(RuntimeValue::List(
                    Rc::new(elems),
                    outcome.unwrap_or(ScalarType::Int),
                ))
            }
            Expression::FunctionCall { name, args } => {
                let func = self
                    .functions
                    .get(&name.value)
                    .ok_or_else(|| RuntimeError::UndefinedFunction {
                        range: expression.range.into(),
                        name: name.value.clone(),
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
            Expression::Reference(name) => eval_context.env.borrow().get(name).ok_or_else(|| {
                RuntimeError::UndefinedReference {
                    range: expression.range.into(),
                    name: name.clone(),
                }
            }),
            Expression::Int(i) => Ok(RuntimeValue::Scalar(ScalarValue::Int(*i))),
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
                RuntimeValue::Scalar(ScalarValue::Int(i)) => {
                    usize::try_from(i.max(0)).expect("converting a positive i32 into usize")
                }
                repeat_value => {
                    return Err(RuntimeError::InvalidRepeatExpression {
                        range: repeat.range.into(),
                        found: repeat_value.runtime_type(),
                        value: repeat_value,
                    })
                }
            },
            None => 1,
        };
        let base = match &item.item {
            BareListItem::Expr(expression) => self.evaluate(eval_context, expression),
            BareListItem::Range(start_expr, end_expr) => {
                let start = self.evaluate(eval_context, start_expr)?;
                let start = match start {
                    RuntimeValue::Scalar(ScalarValue::Int(i)) => i,
                    _ => {
                        return Err(RuntimeError::RangeHasNonSequenceEndpoints {
                            range: start_expr.range.into(),
                            found: start.runtime_type(),
                        })
                    }
                };
                let end = self.evaluate(eval_context, end_expr)?;
                let end = match end {
                    RuntimeValue::Scalar(ScalarValue::Int(i)) => i,
                    _ => {
                        return Err(RuntimeError::RangeHasNonSequenceEndpoints {
                            range: end_expr.range.into(),
                            found: end.runtime_type(),
                        })
                    }
                };
                Ok(RuntimeValue::List(
                    Rc::new((start..=end).map(ScalarValue::Int).collect()),
                    ScalarType::Int,
                ))
            }
        }?;
        let outcome_type = base.outcome_type();
        Ok(RuntimeValue::List(
            Rc::new(base.to_list(repeat_count)),
            outcome_type,
        ))
    }

    fn evaluate_function_call(
        &mut self,
        eval_context: &EvalContext,
        function: &WithRange<Function>,
        args: Vec<WithRange<RuntimeValue>>,
    ) -> Result<RuntimeValue, RuntimeError> {
        if eval_context.recursion_depth >= self.recursion_depth {
            return Ok(vec![].into());
        }

        // First, convert the arguments to the expected types.
        let expected_types = function.value.get_arg_types();
        if expected_types.len() != args.len() {
            panic!(
                "wrong number of arguments; this should have been caught by function name matching"
            );
        }
        let arg_ranges = args.iter().map(|arg| arg.range).collect::<Vec<_>>();
        let args = args
            .into_iter()
            .zip(expected_types.iter())
            .map(|(arg, expected)| coerce_arg(arg.value, expected.as_ref(), arg.range))
            .collect::<Result<Vec<_>, _>>()?;

        // This vector will contain references to pools which match an int or list arguments. These
        // are the pools over which we need to iterate to get the argument values.
        let mut pools = Vec::new();
        // Contains (index, is_int, outcome_type) for the corresponding pool.
        let mut pool_iterator_info = Vec::new();
        for (i, (arg, expected_type)) in args.iter().zip(expected_types.iter()).enumerate() {
            let pool = match arg {
                RuntimeValue::Pool(p, _) => Some(p),
                _ => None,
            };
            if let (Some(p), Some(expected)) = (pool, expected_type) {
                if !matches!(expected.shape, StaticType::Int | StaticType::List) {
                    continue;
                }
                // If the expected type is an Int, `coerce_arg` has already turned the pool into a sum,
                // with outcomes of length 1.
                pools.push(p);
                pool_iterator_info.push((i, expected.shape == StaticType::Int, arg.outcome_type()));
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
            for ((i, is_scalar, outcome_type), value) in
                pool_iterator_info.iter().zip(values.iter())
            {
                if *is_scalar {
                    args[*i] = RuntimeValue::Scalar(value[0].clone());
                } else {
                    args[*i] = RuntimeValue::List(
                        Rc::new(reverse_if(!self.lowest_first, value)),
                        outcome_type.clone(),
                    );
                }
            }
            results.push((
                self.call_function(eval_context, function, &args, &arg_ranges)?,
                weight,
            ));
        }

        // TODO this is similar to the logic in flat_map in Pool, find a way to use that?
        let mut total_results = BTreeMap::<ScalarValue, Rational>::new();
        let mut lcm = Natural::ONE;
        let mut result_type: Option<ScalarType> = None;
        for (result, weight) in results {
            if let RuntimeValue::Pool(ref p, _) = result {
                // The empty die is ignored in this context, but the empty list is not.
                if p.ordered_outcomes().is_empty() {
                    continue;
                }
            }
            let current_type = result.outcome_type();
            merge_outcome_type(
                &mut result_type,
                current_type,
                function.range,
                "function evaluations returned incompatible outcome types",
            )?;
            match &result {
                RuntimeValue::List(values, _) if !result.is_numeric() && values.len() != 1 => {
                    return Err(RuntimeError::EnumTypeError {
                        range: function.range.into(),
                        message:
                            "a non-numeric sequence returned during pool evaluation cannot be summed"
                                .to_string(),
                    });
                }
                _ => {}
            }
            let summed = sum_pool(&result.to_pool());
            let total_count = summed
                .ordered_outcomes()
                .iter()
                .map(|(_, count)| count)
                .sum();
            lcm = lcm.lcm(&total_count);
            for (outcome, count) in summed.ordered_outcomes() {
                *total_results
                    .entry(outcome.clone())
                    .or_insert(Rational::ZERO) +=
                    Rational::from_naturals(count * &weight, total_count.clone());
            }
        }
        let result_type = result_type.unwrap_or(ScalarType::Int);
        Ok(RuntimeValue::Pool(
            Rc::new(
                total_results
                    .into_iter()
                    .map(|(outcome, weight)| {
                        let (numerator, denominator) =
                            (weight * Rational::from(&lcm)).into_numerator_and_denominator();
                        debug_assert_eq!(denominator, Natural::ONE);
                        (outcome, numerator)
                    })
                    .collect::<Pool<ScalarValue>>(),
            ),
            result_type,
        ))
    }

    fn call_function(
        &mut self,
        eval_context: &EvalContext,
        function: &WithRange<Function>,
        args: &[RuntimeValue],
        arg_ranges: &[ast::Range],
    ) -> Result<RuntimeValue, RuntimeError> {
        match &function.value {
            Function::Primitive(primitive) => (primitive.execute)(
                args,
                arg_ranges,
                self.explode_depth,
                self.lowest_first,
                function.range,
            ),
            Function::UserDefined(user_function) => {
                let mut new_env = ValEnv::with_parent(Rc::clone(&eval_context.env));
                for (arg, formal) in args.iter().zip(user_function.definition.args.iter()) {
                    new_env.insert(formal.value.name.clone(), arg.clone());
                }
                let new_context = EvalContext {
                    env: Rc::new(RefCell::new(new_env)),
                    recursion_depth: eval_context.recursion_depth + 1,
                    block_depth: eval_context.block_depth + 1,
                };
                let mut result = None;
                for statement in &user_function.definition.body {
                    result = self.execute_statement(&new_context, statement)?;
                    if result.is_some() {
                        break;
                    }
                }
                // If there's no result, there was no return statement in the function.
                Ok(result.unwrap_or(Pool::from_list(1, vec![]).into()))
            }
        }
    }
}

fn apply_unary_op(
    op: &WithRange<UnaryOp>,
    operand: &RuntimeValue,
) -> Result<RuntimeValue, RuntimeError> {
    match op.value {
        UnaryOp::D => make_d(None, operand, op.range),
        UnaryOp::Negate | UnaryOp::Invert if !operand.is_numeric() => {
            Err(RuntimeError::EnumTypeError {
                range: op.range.into(),
                message: format!("operator {} is not defined for this scalar type", op.value),
            })
        }
        UnaryOp::Negate => Ok(operand.map_numeric_outcomes(|o| -o)),
        UnaryOp::Invert => Ok(operand.map_numeric_outcomes(|o| if o == 0 { 1 } else { 0 })),
        UnaryOp::Length => Ok(match operand {
            RuntimeValue::Scalar(ScalarValue::Int(i)) => i32::try_from(i.abs().to_string().len())
                .expect("vector length fits in i32")
                .into(),
            RuntimeValue::Scalar(ScalarValue::Enum { .. }) => 1.into(),
            RuntimeValue::Scalar(ScalarValue::Tuple(fields)) => i32::try_from(fields.len())
                .expect("tuple length fits in i32")
                .into(),
            RuntimeValue::List(list, _) => i32::try_from(list.len())
                .expect("vector length fits in i32")
                .into(),
            RuntimeValue::Pool(d, _) => i32::try_from(d.dimension())
                .expect("vector length fits in i32")
                .into(),
        }),
    }
}

fn apply_binary_op(
    op: &WithRange<BinaryOp>,
    left: &RuntimeValue,
    left_range: ast::Range,
    right: &RuntimeValue,
    lowest_first: bool,
) -> Result<RuntimeValue, RuntimeError> {
    let has_non_numeric = !left.is_numeric() || !right.is_numeric();
    if has_non_numeric
        && !matches!(
            op.value,
            BinaryOp::D | BinaryOp::At | BinaryOp::Eq | BinaryOp::Ne
        )
    {
        return Err(RuntimeError::EnumTypeError {
            range: op.range.into(),
            message: format!(
                "operator {} is not defined for these scalar types",
                op.value
            ),
        });
    }
    if matches!(op.value, BinaryOp::Eq | BinaryOp::Ne) && !left.has_same_outcome_type(right) {
        return Err(RuntimeError::EnumTypeError {
            range: op.range.into(),
            message: "equality requires operands with the same outcome type".to_string(),
        });
    }
    match &op.value {
        BinaryOp::D => make_d(Some(left), right, op.range),
        BinaryOp::At => {
            if !left.is_numeric() {
                return Err(RuntimeError::EnumTypeError {
                    range: op.range.into(),
                    message: "only integers can be used as positions".to_string(),
                });
            }
            let left = match left {
                RuntimeValue::Scalar(ScalarValue::Int(i)) => Rc::new(vec![*i]),
                RuntimeValue::List(lst, _) => {
                    Rc::new(lst.iter().map(expect_int).collect::<Vec<_>>())
                }
                RuntimeValue::Pool(_, _) => {
                    return Err(RuntimeError::InvalidArgumentToOperator {
                        operator_range: op.range.into(),
                        op: op.value,
                        expected: "an int or a list",
                        found_range: left_range.into(),
                        found: left.runtime_type(),
                    })
                }
                RuntimeValue::Scalar(ScalarValue::Enum { .. } | ScalarValue::Tuple(_)) => {
                    unreachable!()
                }
            };
            match right {
                RuntimeValue::Scalar(ScalarValue::Int(i)) => {
                    let digits: Vec<i32> = i
                        .abs()
                        .to_string()
                        .chars()
                        // Unwrap here is ok as all chars for a *positive* integer are valid digits.
                        .map(|c| {
                            i32::try_from(c.to_digit(10).unwrap()).expect("digit fits in i32")
                                * i.signum()
                        })
                        .collect();
                    Ok(select_positions(&left, &digits, lowest_first).into())
                }
                RuntimeValue::List(lst, _) if right.is_numeric() => Ok(select_positions(
                    &left,
                    &lst.iter().map(expect_int).collect::<Vec<_>>(),
                    false,
                )
                .into()),
                RuntimeValue::Pool(p, _) if right.is_numeric() => Ok(RuntimeValue::Pool(
                    Rc::new(select_in_dice(&left, p, lowest_first)),
                    ScalarType::Int,
                )),
                RuntimeValue::List(lst, _) => {
                    if left.len() != 1 {
                        return Err(RuntimeError::EnumTypeError { range: op.range.into(), message: "selecting multiple non-numeric positions would require summing their values".to_string() });
                    }
                    let index = left[0];
                    if index < 1 || index > i32::try_from(lst.len()).unwrap_or(i32::MAX) {
                        return Err(RuntimeError::EnumTypeError {
                            range: op.range.into(),
                            message: "position is out of range".to_string(),
                        });
                    }
                    Ok(RuntimeValue::Scalar(
                        lst[usize::try_from(index - 1).unwrap()].clone(),
                    ))
                }
                RuntimeValue::Scalar(ScalarValue::Enum { .. } | ScalarValue::Tuple(_))
                | RuntimeValue::Pool(_, _) => Err(RuntimeError::EnumTypeError {
                    range: op.range.into(),
                    message: "positional selection is not defined for this value".to_string(),
                }),
            }
        }
        BinaryOp::Pow => math_binary_op(left, right, |a, b| {
            if b < 0 {
                Err(format!("Cannot raise {} to negative power {}", a, b))
            } else {
                a.checked_pow(b.unsigned_abs())
                    .ok_or_else(|| format!("Power overflow: {} ^ {}", a, b))
            }
        })
        .map_err(|msg| RuntimeError::MathError {
            range: op.range.into(),
            message: msg,
        }),
        BinaryOp::Add => math_binary_op(left, right, |a, b| {
            a.checked_add(b)
                .ok_or_else(|| format!("Addition overflow: {} + {}", a, b))
        })
        .map_err(|msg| RuntimeError::MathError {
            range: op.range.into(),
            message: msg,
        }),
        BinaryOp::Sub => math_binary_op(left, right, |a, b| {
            a.checked_sub(b)
                .ok_or_else(|| format!("Subtraction overflow: {} - {}", a, b))
        })
        .map_err(|msg| RuntimeError::MathError {
            range: op.range.into(),
            message: msg,
        }),
        BinaryOp::Mul => math_binary_op(left, right, |a, b| {
            a.checked_mul(b)
                .ok_or_else(|| format!("Multiplication overflow: {} * {}", a, b))
        })
        .map_err(|msg| RuntimeError::MathError {
            range: op.range.into(),
            message: msg,
        }),
        BinaryOp::Div => math_binary_op(left, right, |a, b| {
            if b == 0 {
                Err(format!("Cannot divide {} by zero", a))
            } else {
                Ok(a / b)
            }
        })
        .map_err(|msg| RuntimeError::MathError {
            range: op.range.into(),
            message: msg,
        }),
        BinaryOp::Eq => Ok(equality_binary_op(left, right, true)),
        BinaryOp::Ne => Ok(equality_binary_op(left, right, false)),
        BinaryOp::Lt => Ok(comp_binary_op(
            left,
            right,
            |a, b| if a < b { 1 } else { 0 },
            |a, b| if a < b { 1 } else { 0 },
        )),
        BinaryOp::Le => Ok(comp_binary_op(
            left,
            right,
            |a, b| if a <= b { 1 } else { 0 },
            |a, b| if a <= b { 1 } else { 0 },
        )),
        BinaryOp::Gt => Ok(comp_binary_op(
            left,
            right,
            |a, b| if a > b { 1 } else { 0 },
            |a, b| if a > b { 1 } else { 0 },
        )),
        BinaryOp::Ge => Ok(comp_binary_op(
            left,
            right,
            |a, b| if a >= b { 1 } else { 0 },
            |a, b| if a >= b { 1 } else { 0 },
        )),
        BinaryOp::Or => {
            math_binary_op(left, right, |a, b| Ok(if a != 0 || b != 0 { 1 } else { 0 })).map_err(
                |msg| RuntimeError::MathError {
                    range: op.range.into(),
                    message: msg,
                },
            )
        }
        BinaryOp::And => {
            math_binary_op(left, right, |a, b| Ok(if a != 0 && b != 0 { 1 } else { 0 })).map_err(
                |msg| RuntimeError::MathError {
                    range: op.range.into(),
                    message: msg,
                },
            )
        }
    }
}

fn select_positions(indices: &[i32], vec: &[i32], lowest_first: bool) -> i32 {
    indices
        .iter()
        .map(|&i| {
            if i < 1 || i > i32::try_from(vec.len()).expect("vector length fits in i32") {
                return 0;
            }
            let i = usize::try_from(i)
                .expect("i is positive, and a positive i32 should fit in a usize");
            let i = if lowest_first { vec.len() - i } else { i - 1 };
            vec.get(i).copied().unwrap_or(0)
        })
        .sum()
}

fn select_in_dice(
    indices: &[i32],
    pool: &Pool<ScalarValue>,
    lowest_first: bool,
) -> Pool<ScalarValue> {
    let dimension = usize::try_from(pool.dimension()).expect("usize is at least 32 bits");
    let mut keep_list = vec![false; dimension];
    for &i in indices {
        if i < 1 || i > i32::try_from(dimension).expect("vector length fits in i32") {
            continue;
        }
        let i =
            usize::try_from(i).expect("i is positive, and a positive i32 should fit in a usize");
        let i = if !lowest_first { dimension - i } else { i - 1 };
        keep_list[i] = true;
    }
    pool.clone()
        .map_outcomes(|outcome| expect_int(&outcome))
        .sum_with_keep_list(&keep_list)
        .map_outcomes(ScalarValue::Int)
}

fn math_binary_op(
    left: &RuntimeValue,
    right: &RuntimeValue,
    f: impl Fn(i32, i32) -> Result<i32, String>,
) -> Result<RuntimeValue, String> {
    let left = flatten_list(left);
    let right = flatten_list(right);
    let (left_pool, right_pool) = match (left, right) {
        (DLeftSide::Int(a), DLeftSide::Int(b)) => return Ok(f(a, b)?.into()),
        (left, right) => (left.to_pool().sum(), right.to_pool().sum()),
    };

    // For pool operations, we need to handle errors during mapping
    let mut results = Vec::new();
    for (left_outcome, left_weight) in left_pool.ordered_outcomes() {
        for (right_outcome, right_weight) in right_pool.ordered_outcomes() {
            match f(*left_outcome, *right_outcome) {
                Ok(result) => results.push((result, left_weight * right_weight)),
                Err(err) => return Err(err),
            }
        }
    }

    let mut pool_map = std::collections::BTreeMap::new();
    for (outcome, weight) in results {
        *pool_map.entry(outcome).or_insert(Natural::ZERO) += weight;
    }

    Ok(Pool::from(pool_map.into_iter().collect::<Vec<_>>()).into())
}

/// Binary ops behave differently depending on the types of their arguments.
/// (int, int) is the straightforward case
/// (list, list) is in lexicographic order
/// (list, int) or (int, list) are elementwise then summed (counted)
/// (pool, pool) sums both sides, then applies the binary op elementwise
fn comp_binary_op(
    left: &RuntimeValue,
    right: &RuntimeValue,
    int_comp: impl Fn(i32, i32) -> i32,
    list_comp: impl Fn(&[i32], &[i32]) -> i32,
) -> RuntimeValue {
    match (left, right) {
        (RuntimeValue::Scalar(ScalarValue::Int(a)), RuntimeValue::Scalar(ScalarValue::Int(b))) => {
            int_comp(*a, *b).into()
        }
        (RuntimeValue::List(a, _), RuntimeValue::List(b, _)) => list_comp(
            &a.iter().map(expect_int).collect::<Vec<_>>(),
            &b.iter().map(expect_int).collect::<Vec<_>>(),
        )
        .into(),
        (RuntimeValue::List(a, _), RuntimeValue::Scalar(ScalarValue::Int(b))) => a
            .iter()
            .map(|a| int_comp(expect_int(a), *b))
            .sum::<i32>()
            .into(),
        (RuntimeValue::Scalar(ScalarValue::Int(a)), RuntimeValue::List(b, _)) => b
            .iter()
            .map(|b| int_comp(*a, expect_int(b)))
            .sum::<i32>()
            .into(),
        _ => {
            // At least one is a pool
            let left_pool = sum_pool(&left.to_pool());
            let right_pool = sum_pool(&right.to_pool());
            RuntimeValue::Pool(
                Rc::new(left_pool.flat_map(|left_outcome| {
                    right_pool
                        .clone()
                        .map_outcomes(|right_outcome| {
                            ScalarValue::Int(int_comp(
                                expect_int(&left_outcome[0]),
                                expect_int(&right_outcome),
                            ))
                        })
                        .into()
                })),
                ScalarType::Int,
            )
        }
    }
}

fn equality_binary_op(left: &RuntimeValue, right: &RuntimeValue, equal: bool) -> RuntimeValue {
    let compare = |a: &ScalarValue, b: &ScalarValue| i32::from((a == b) == equal);
    match (left, right) {
        (RuntimeValue::Scalar(a), RuntimeValue::Scalar(b)) => compare(a, b).into(),
        (RuntimeValue::List(a, _), RuntimeValue::List(b, _)) => i32::from((a == b) == equal).into(),
        (RuntimeValue::List(a, _), RuntimeValue::Scalar(b)) => {
            a.iter().map(|a| compare(a, b)).sum::<i32>().into()
        }
        (RuntimeValue::Scalar(a), RuntimeValue::List(b, _)) => {
            b.iter().map(|b| compare(a, b)).sum::<i32>().into()
        }
        _ => {
            let left_pool = sum_pool(&left.to_pool());
            let right_pool = sum_pool(&right.to_pool());
            RuntimeValue::Pool(
                Rc::new(left_pool.flat_map(|left_outcome| {
                    right_pool
                        .clone()
                        .map_outcomes(|right_outcome| {
                            ScalarValue::Int(compare(&left_outcome[0], &right_outcome))
                        })
                        .into()
                })),
                ScalarType::Int,
            )
        }
    }
}

enum DLeftSide {
    Int(i32),
    Pool(Rc<Pool<i32>>),
}

impl DLeftSide {
    fn to_pool(&self) -> Rc<Pool> {
        match self {
            DLeftSide::Int(i) => Rc::new(Pool::from_list(1, vec![*i])),
            DLeftSide::Pool(p) => Rc::clone(p),
        }
    }
}

enum DRightSide {
    List(Vec<ScalarValue>),
    Pool(Rc<Pool<ScalarValue>>),
}

fn flatten_list(arg: &RuntimeValue) -> DLeftSide {
    match arg {
        RuntimeValue::Scalar(ScalarValue::Int(i)) => DLeftSide::Int(*i),
        RuntimeValue::List(list, _) => DLeftSide::Int(list.iter().map(expect_int).sum()),
        RuntimeValue::Pool(pool, _) => DLeftSide::Pool(Rc::new(
            (**pool)
                .clone()
                .map_outcomes(|outcome| expect_int(&outcome)),
        )),
        RuntimeValue::Scalar(ScalarValue::Enum { .. } | ScalarValue::Tuple(_)) => {
            unreachable!("non-numeric values are rejected before numeric operations")
        }
    }
}

/// Executes the unary or binary version of the _n_ `d` _m_ operator.
fn make_d(
    left: Option<&RuntimeValue>,
    right: &RuntimeValue,
    range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    if left.is_some_and(|value| !value.is_numeric()) {
        return Err(RuntimeError::EnumTypeError {
            range: range.into(),
            message: "only numeric values can be used as dice counts".to_string(),
        });
    }
    let repeat = left.map_or(DLeftSide::Int(1), flatten_list);
    let outcome_type = right.outcome_type();
    let right = match right {
        RuntimeValue::Scalar(ScalarValue::Int(sides)) => {
            if *sides > 0 {
                DRightSide::List((1..=*sides).map(ScalarValue::Int).collect())
            } else if *sides == 0 {
                DRightSide::List(vec![ScalarValue::Int(0)])
            } else {
                DRightSide::List((*sides..=-1).map(ScalarValue::Int).collect())
            }
        }
        RuntimeValue::List(list, _) => DRightSide::List((**list).clone()),
        RuntimeValue::Pool(d, _) => DRightSide::Pool(Rc::clone(d)),
        RuntimeValue::Scalar(ScalarValue::Enum { .. } | ScalarValue::Tuple(_)) => {
            return Err(RuntimeError::EnumTypeError {
                range: range.into(),
                message: "a non-numeric scalar cannot specify die sides; use a sequence"
                    .to_string(),
            })
        }
    };
    if outcome_type != ScalarType::Int && !matches!(&repeat, DLeftSide::Int(1)) {
        return Err(RuntimeError::EnumTypeError {
            range: range.into(),
            message: "non-numeric pools must have dimension one".to_string(),
        });
    }
    let result: RuntimeValue = match (repeat, right) {
        (DLeftSide::Int(i), DRightSide::List(list)) => {
            RuntimeValue::Pool(Rc::new(make_pool(i, list)), outcome_type.clone())
        }
        (DLeftSide::Int(i), DRightSide::Pool(p)) => {
            let mut new_pool = (*p).clone();
            if i < 0 {
                new_pool = new_pool.map_outcomes(|outcome| ScalarValue::Int(-expect_int(&outcome)));
            }
            new_pool.set_dimension(new_pool.dimension() * i.unsigned_abs());
            RuntimeValue::Pool(Rc::new(new_pool), outcome_type.clone())
        }
        (DLeftSide::Pool(left_p), right) => {
            let left_p = (*left_p).sum();
            let right = match right {
                DRightSide::List(list) => Pool::from_list(1, list),
                DRightSide::Pool(p) => sum_pool(&p),
            };
            // At this point both |left_p| and |right| have a count of 1.
            // For each outcome in the left pool, sum the right pool with itself k times
            RuntimeValue::Pool(
                Rc::new(left_p.flat_map(|count| {
                    debug_assert_eq!(
                        count.len(),
                        1,
                        "summed distribution has count of {}",
                        count.len()
                    );
                    let mut dup_right = right.clone();
                    let multiplier = count[0];
                    if multiplier < 0 {
                        dup_right = dup_right
                            .map_outcomes(|outcome| ScalarValue::Int(-expect_int(&outcome)));
                    }
                    dup_right.set_dimension(dup_right.dimension() * multiplier.unsigned_abs());
                    sum_pool(&dup_right).into()
                })),
                ScalarType::Int,
            )
        }
    };
    Ok(result)
}

fn make_pool(mut n: i32, mut sides: Vec<ScalarValue>) -> Pool<ScalarValue> {
    if n < 0 {
        for side in sides.iter_mut() {
            *side = ScalarValue::Int(-expect_int(side));
        }
        n = -n;
    }
    Pool::from_list(u32::try_from(n).expect("n is positive"), sides)
}

fn merge_outcome_type(
    accumulated: &mut Option<ScalarType>,
    next: ScalarType,
    range: ast::Range,
    error_message: &'static str,
) -> Result<(), RuntimeError> {
    let merged = match accumulated.as_ref() {
        None => next,
        Some(current) => current
            .merged_with(&next)
            .ok_or_else(|| RuntimeError::EnumTypeError {
                range: range.into(),
                message: error_message.to_string(),
            })?,
    };
    *accumulated = Some(merged);
    Ok(())
}

fn coerce_arg(
    arg: RuntimeValue,
    expected: Option<&ResolvedArgType>,
    range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    let Some(expected) = expected else {
        return Ok(arg);
    };
    let actual_outcome = arg.outcome_type();
    if let Some(required) = &expected.outcome {
        if actual_outcome.merged_with(required).is_none() {
            return Err(RuntimeError::EnumTypeError {
                range: range.into(),
                message: format!(
                    "expected {}, found {}",
                    required.display_name(),
                    actual_outcome.display_name()
                ),
            });
        }
    }
    match (arg, expected.shape) {
        (value @ RuntimeValue::Scalar(_), StaticType::Int) => Ok(value),
        (RuntimeValue::Scalar(value), StaticType::List) => {
            let outcome_type = value.scalar_type();
            Ok(RuntimeValue::List(Rc::new(vec![value]), outcome_type))
        }
        (RuntimeValue::Scalar(value), StaticType::Pool) => {
            let outcome_type = value.scalar_type();
            Ok(RuntimeValue::Pool(
                Rc::new(Pool::from_list(1, vec![value])),
                outcome_type,
            ))
        }
        (RuntimeValue::List(list, _), StaticType::Int) if actual_outcome == ScalarType::Int => Ok(
            RuntimeValue::Scalar(ScalarValue::Int(list.iter().map(expect_int).sum())),
        ),
        (RuntimeValue::List(_, _), StaticType::Int) => Err(RuntimeError::EnumTypeError {
            range: range.into(),
            message: "a non-numeric sequence cannot be summed into a scalar".to_string(),
        }),
        (value @ RuntimeValue::List(_, _), StaticType::List) => Ok(value),
        (RuntimeValue::List(list, outcome_type), StaticType::Pool) => Ok(RuntimeValue::Pool(
            Rc::new(Pool::from_list(1, (*list).clone())),
            outcome_type,
        )),
        (RuntimeValue::Pool(pool, outcome_type), StaticType::Int) => {
            Ok(RuntimeValue::Pool(Rc::new(sum_pool(&pool)), outcome_type))
        }
        (value @ RuntimeValue::Pool(_, _), _) => Ok(value),
    }
}

impl From<ast::Range> for SourceSpan {
    fn from(range: ast::Range) -> Self {
        SourceSpan::new(range.start.into(), range.end - range.start)
    }
}

fn interpolate_variable_names(
    template: &WithRange<String>,
    vars: &RcValEnv,
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
            if !valid_end || name.is_empty() || name.chars().any(|c| !c.is_ascii_uppercase()) {
                result.push('[');
                result.push_str(&name);
                if valid_end {
                    result.push(']');
                }
            } else if let Some(value) = vars.borrow().get(&name) {
                result.push_str(&value.to_string());
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

fn reverse_if<T: Clone>(should_reverse: bool, v: &[T]) -> Vec<T> {
    if should_reverse {
        v.iter().cloned().rev().collect()
    } else {
        v.into()
    }
}

#[derive(Debug, Error, Diagnostic)]
#[error("Runtime error")]
pub enum RuntimeError {
    #[error("Enum type error: {message}")]
    EnumTypeError {
        #[label = "{message}"]
        range: SourceSpan,
        message: String,
    },

    #[error("Output statement inside a function")]
    #[diagnostic(help("Output statements can only appear outside functions."))]
    OutputNotAtTopLevel {
        #[label = "Output statement inside a function"]
        range: SourceSpan,
    },

    #[error("Set statement inside a function")]
    #[diagnostic(help("Set statements can only appear outside functions."))]
    SetNotAtTopLevel {
        #[label = "Set statement inside a function"]
        range: SourceSpan,
    },

    #[error("Return statxement outside a function")]
    ReturnOutsideFunction {
        #[label = "Return statement outside a function"]
        range: SourceSpan,
    },

    #[error("Loops must iterate over sequences")]
    LoopOverNonSequence {
        #[label = "This is a {found}."]
        range: SourceSpan,
        found: StaticType,
    },

    #[error("Reference to undefined to variable [{name}]")]
    UndefinedReference {
        #[label = "Variable not defined"]
        range: SourceSpan,
        name: String,
    },

    #[error("Reference to undefined function [{name}]")]
    UndefinedFunction {
        #[label = "No function named [{name}]"]
        range: SourceSpan,
        name: String,
    },

    #[error("Conditions to `if` statements must be numbers.")]
    InvalidCondition {
        #[label = "This is a {found} with value {value}."]
        range: SourceSpan,
        found: StaticType,
        value: RuntimeValue,
    },

    #[error("Both sides of a range constructor must evaluate to numbers.")]
    RangeHasNonSequenceEndpoints {
        #[label = "This is a {found}."]
        range: SourceSpan,
        found: StaticType,
    },

    #[error("Invalid argument to operator")]
    InvalidArgumentToOperator {
        #[label = "Operator {op} expects {expected}."]
        operator_range: SourceSpan,
        op: BinaryOp,
        expected: &'static str,

        // TODO - show value
        #[label = "This is a {found}."]
        found_range: SourceSpan,
        found: StaticType,
    },

    #[error("Invalid argument to function")]
    NegativeArgumentToFunction {
        #[label = "Function [{name}] expects a nonnegative integer."]
        range: SourceSpan,
        name: String,

        #[label = "This evaluates to {value}."]
        found_range: SourceSpan,
        value: i32,
    },

    #[error("Invalid repeat expression")]
    #[diagnostic(help("The expression inside the repeat operator must evaluate to an int."))]
    InvalidRepeatExpression {
        #[label = "This is a {found}: {value}."]
        range: SourceSpan,
        found: StaticType,
        value: RuntimeValue,
    },

    #[error("A mathematical error occurred.")]
    MathError {
        #[label = "{message}"]
        range: SourceSpan,
        message: String,
    },
}

impl RuntimeError {
    pub fn range(&self) -> ast::Range {
        match self {
            RuntimeError::EnumTypeError { range, .. } => range.into(),
            RuntimeError::OutputNotAtTopLevel { range } => range.into(),
            RuntimeError::SetNotAtTopLevel { range } => range.into(),
            RuntimeError::ReturnOutsideFunction { range } => range.into(),
            RuntimeError::LoopOverNonSequence { range, .. } => range.into(),
            RuntimeError::UndefinedReference { range, .. } => range.into(),
            RuntimeError::UndefinedFunction { range, .. } => range.into(),
            RuntimeError::InvalidCondition { range, .. } => range.into(),
            RuntimeError::RangeHasNonSequenceEndpoints { range, .. } => range.into(),
            RuntimeError::InvalidArgumentToOperator { operator_range, .. } => operator_range.into(),
            RuntimeError::NegativeArgumentToFunction { range, .. } => range.into(),
            RuntimeError::InvalidRepeatExpression { range, .. } => range.into(),
            RuntimeError::MathError { range, .. } => range.into(),
        }
    }
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

        let env = Rc::new(RefCell::new(ValEnv::new()));
        env.borrow_mut().insert("A".to_string(), 1.into());
        env.borrow_mut().insert("B".to_string(), 2.into());
        env.borrow_mut().insert("C".to_string(), 3.into());
        assert_eq!(
            interpolate_variable_names(&ranged("A + B = [A] + [B]"), &env).unwrap(),
            "A + B = 1 + 2"
        );
        assert!(
            interpolate_variable_names(&ranged("A + B = [A] + [B] + [INVALID]"), &env).is_err()
        );
        assert_eq!(
            interpolate_variable_names(&ranged("A + B = [A] + [B] + [invalid]"), &env).unwrap(),
            "A + B = 1 + 2 + [invalid]"
        );
        assert_eq!(
            interpolate_variable_names(&ranged("A + B = [A] + [B] + [C"), &env).unwrap(),
            "A + B = 1 + 2 + [C"
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

    #[test]
    fn test_valid_math_operations() {
        use crate::ast::{BinaryOp, Range};

        // Test valid division
        let left = 10.into();
        let right = 2.into();
        let op = WithRange {
            value: BinaryOp::Div,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Ok(RuntimeValue::Scalar(ScalarValue::Int(5))) => {}
            _ => panic!("Expected successful division result of 5"),
        }

        // Test valid exponentiation
        let left = 2.into();
        let right = 3.into();
        let op = WithRange {
            value: BinaryOp::Pow,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Ok(RuntimeValue::Scalar(ScalarValue::Int(8))) => {}
            _ => panic!("Expected successful exponentiation result of 8"),
        }
    }

    #[test]
    fn test_addition_overflow() {
        use crate::ast::{BinaryOp, Range};

        let left = i32::MAX.into();
        let right = 1.into();
        let op = WithRange {
            value: BinaryOp::Add,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Addition overflow"));
                assert!(message.contains(&format!("{}", i32::MAX)));
            }
            _ => panic!("Expected MathError for addition overflow"),
        }
    }

    #[test]
    fn test_subtraction_overflow() {
        use crate::ast::{BinaryOp, Range};

        let left = i32::MIN.into();
        let right = 1.into();
        let op = WithRange {
            value: BinaryOp::Sub,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Subtraction overflow"));
                assert!(message.contains(&format!("{}", i32::MIN)));
            }
            _ => panic!("Expected MathError for subtraction overflow"),
        }
    }

    #[test]
    fn test_multiplication_overflow() {
        use crate::ast::{BinaryOp, Range};

        let left = i32::MAX.into();
        let right = 2.into();
        let op = WithRange {
            value: BinaryOp::Mul,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Multiplication overflow"));
                assert!(message.contains(&format!("{}", i32::MAX)));
            }
            _ => panic!("Expected MathError for multiplication overflow"),
        }
    }

    #[test]
    fn test_power_overflow() {
        use crate::ast::{BinaryOp, Range};

        let left = 2.into();
        let right = 32.into(); // 2^32 overflows i32
        let op = WithRange {
            value: BinaryOp::Pow,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Power overflow"));
                assert!(message.contains("2 ^ 32"));
            }
            _ => panic!("Expected MathError for power overflow"),
        }
    }
}
