use std::{collections::HashMap, rc::Rc};

use malachite::Natural;
use serde::Serialize;

use crate::{
    ast::{self, StaticType},
    dice::{Pool, explode, reroll},
    error::{
        PrimitiveArgumentError, PrimitiveArgumentErrorKind, PrimitiveArgumentsError,
        PrimitiveValueError,
    },
    eval::{ElementType, ElementValue, Function, RuntimeError, RuntimeValue, sum_pool},
};

/// Evaluation context passed to every primitive, bundling the ambient settings
/// and per-call metadata that most primitives don't need but a few do.
#[derive(Debug, Clone, Copy)]
pub struct PrimitiveCtx<'a> {
    /// Source ranges of the arguments, parallel to `args`.
    pub arg_ranges: &'a [ast::Range],
    /// The configured `set "explode depth"`.
    pub explode_depth: usize,
    /// Whether positional order is lowest-first (`set "position order"`).
    pub lowest_first: bool,
    /// Source range of the whole function call, used for error reporting.
    pub function_range: ast::Range,
    /// Canonical function identifier from the primitive registry.
    pub identifier: &'static str,
}

type PrimitiveExecutor =
    fn(&[RuntimeValue], PrimitiveCtx) -> Result<RuntimeValue, crate::eval::RuntimeError>;

#[derive(Debug)]
pub struct Primitive {
    pub identifier: &'static str,
    pub arg_types: &'static [Option<StaticType>],
    pub accepts_non_numeric: bool,
    pub execute: PrimitiveExecutor,
}

/// Presentation metadata for a built-in function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PrimitiveMetadata {
    /// The evaluator's canonical function identifier, with `{}` argument slots.
    pub identifier: &'static str,
    /// A typed signature intended for display to users.
    pub signature: &'static str,
    /// A CodeMirror-compatible snippet body, without surrounding brackets.
    pub snippet: &'static str,
    /// A concise description suitable for completion UI.
    pub documentation: &'static str,
    /// A stable link to the detailed language specification.
    pub documentation_url: &'static str,
}

#[derive(Debug, Clone, Copy)]
enum KeepMode {
    Highest,
    Lowest,
    Middle,
}

impl KeepMode {
    fn name(self) -> &'static str {
        match self {
            Self::Highest => "highest",
            Self::Lowest => "lowest",
            Self::Middle => "middle",
        }
    }
}

/// The argument names of a primitive, in order, read from the signature the
/// registry displays to users.
///
/// Executors report arguments by name, and users see those names in the
/// signature, so both read the same literal rather than repeating it. The
/// registry consistency test checks that every primitive names every argument.
fn argument_names(identifier: &str) -> impl Iterator<Item = &'static str> {
    primitive_signature(identifier)
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|token| token.split_once(':'))
        .map(|(name, _)| name)
}

fn argument_name(identifier: &str, index: usize) -> &'static str {
    argument_names(identifier)
        .nth(index)
        .expect("registry signatures name every argument")
}

fn argument_error(
    ctx: PrimitiveCtx,
    index: usize,
    expected: impl Into<String>,
    value: &RuntimeValue,
) -> PrimitiveArgumentError {
    PrimitiveArgumentError {
        name: argument_name(ctx.identifier, index),
        range: ctx.arg_ranges[index].into(),
        expected: expected.into(),
        value: value.clone(),
    }
}

fn invalid_arguments(
    ctx: PrimitiveCtx,
    requirement: impl Into<String>,
    help: impl Into<String>,
    kind: PrimitiveArgumentErrorKind,
    arguments: Vec<PrimitiveArgumentError>,
) -> RuntimeError {
    RuntimeError::InvalidPrimitiveArguments(Box::new(PrimitiveArgumentsError {
        range: ctx.function_range.into(),
        function: ctx.identifier,
        requirement: requirement.into(),
        help: Some(help.into()),
        kind,
        arguments,
    }))
}

fn absolute_execute(
    args: &[RuntimeValue],
    _ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    let arg = &args[0];
    Ok(arg.map_numeric_outcomes(i32::abs))
}

/// Puts two values on comparable footing for the equality-based primitives.
///
/// Like `=` itself, these primitives are total: values of different kinds are
/// simply never equal, so mismatched outcome types count zero matches rather
/// than failing. The join is consulted only to give the empty sum a concrete
/// type; where there is no join there is nothing to reconcile.
fn materialize_compatible_pair(
    left: &RuntimeValue,
    right: &RuntimeValue,
) -> (RuntimeValue, RuntimeValue) {
    match left.merged_outcome_type(right) {
        Some(outcome_type) => {
            let outcome_type = outcome_type.summed_type();
            (
                left.materialize_identities(&outcome_type),
                right.materialize_identities(&outcome_type),
            )
        }
        None => (left.clone(), right.clone()),
    }
}

fn contains_execute(
    args: &[RuntimeValue],
    _ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    let (haystack, needle) = materialize_compatible_pair(&args[0], &args[1]);
    let RuntimeValue::Element(needle) = needle else {
        unreachable!("contains needle shape is enforced by the evaluator")
    };
    if let RuntimeValue::Pool(haystack, _) = haystack {
        return Ok(haystack.contains(&needle).into());
    }
    let result = match haystack {
        RuntimeValue::Element(haystack) => haystack == needle,
        RuntimeValue::List(haystack, _) => haystack.iter().any(|value| value == &needle),
        RuntimeValue::Pool(_, _) => unreachable!("pool haystack returned above"),
    };
    Ok(if result { 1.into() } else { 0.into() })
}

fn count_execute(
    args: &[RuntimeValue],
    _ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    let (needles, haystack) = materialize_compatible_pair(&args[0], &args[1]);
    let RuntimeValue::List(needles, _) = needles else {
        unreachable!("count needle shape is enforced by the evaluator")
    };
    if let RuntimeValue::Pool(haystack, _) = haystack {
        return Ok(haystack.count(&needles).into());
    }
    let mut needle_map = HashMap::new();
    for needle in needles.iter() {
        *needle_map.entry(needle).or_insert(0) += 1;
    }
    let count = match haystack {
        RuntimeValue::Element(haystack) => needle_map.get(&haystack).copied().unwrap_or(0),
        RuntimeValue::List(haystack, _) => haystack
            .iter()
            .map(|item| needle_map.get(item).copied().unwrap_or(0))
            .sum(),
        RuntimeValue::Pool(_, _) => unreachable!("pool haystack returned above"),
    };
    Ok(count.into())
}

type DieTransform = fn(Vec<(i32, Natural)>, &[i32], usize) -> Vec<(i32, Natural)>;

/// Shared body of the explode/reroll primitives. `on` is the sequence argument
/// naming the triggering faces; `None` (the bare `explode`/`reroll` forms) uses
/// the pool's highest face.
fn transform_die(
    args: &[RuntimeValue],
    on: Option<&RuntimeValue>,
    ctx: PrimitiveCtx,
    transform: DieTransform,
    name: &str,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    let RuntimeValue::Pool(d, _) = &args[0] else {
        panic!("wrong argument types to [{name}]");
    };
    if d.is_empty() {
        return Ok(args[0].clone());
    }
    let die = numeric_pool(d).sum().ordered_outcomes().to_vec();
    let on = match on {
        None => vec![die.last().unwrap().0],
        Some(RuntimeValue::List(cond, _)) => numeric_list(cond),
        Some(_) => panic!("wrong argument types to [{name}]"),
    };
    Ok(Pool::from(transform(die, &on, ctx.explode_depth)).into())
}

fn explode_execute(
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    transform_die(args, None, ctx, explode, "explode")
}

fn explode_on_execute(
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    transform_die(args, Some(&args[1]), ctx, explode, "explode on")
}

fn reroll_execute(
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    transform_die(args, None, ctx, reroll, "reroll")
}

fn reroll_on_execute(
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    transform_die(args, Some(&args[1]), ctx, reroll, "reroll on")
}

/// Shared body of `highest {} of {}`, `lowest {} of {}` and `middle {} of {}`:
/// keep `i` dice from pool `d` according to `mode`, then sum.
fn keep_execute(
    mode: KeepMode,
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    let (RuntimeValue::Element(ElementValue::Int(i)), RuntimeValue::Pool(d, _)) =
        (&args[0], &args[1])
    else {
        panic!("wrong argument types to [{}]", mode.name());
    };
    let keep_list = keep_list_for_primitive(
        mode,
        ctx.identifier,
        *i,
        ctx.arg_ranges[0],
        d.dimension() as usize,
        ctx.function_range,
    )?;
    Ok(numeric_pool(d).sum_with_keep_list(&keep_list).into())
}

fn highest_execute(
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    keep_execute(KeepMode::Highest, args, ctx)
}

fn lowest_execute(
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    keep_execute(KeepMode::Lowest, args, ctx)
}

fn middle_execute(
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    keep_execute(KeepMode::Middle, args, ctx)
}

#[derive(Debug, Clone, Copy)]
enum ExtremumMode {
    Highest,
    Lowest,
}

fn extremum_execute(
    mode: ExtremumMode,
    args: &[RuntimeValue],
    _ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (
        RuntimeValue::Element(ElementValue::Int(i)),
        RuntimeValue::Element(ElementValue::Int(j)),
    ) = (&args[0], &args[1])
    {
        let value = match mode {
            ExtremumMode::Highest => i.max(j),
            ExtremumMode::Lowest => i.min(j),
        };
        Ok((*value).into())
    } else {
        panic!("wrong argument types to extremum primitive");
    }
}

fn highest_of_execute(
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    extremum_execute(ExtremumMode::Highest, args, ctx)
}

fn lowest_of_execute(
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    extremum_execute(ExtremumMode::Lowest, args, ctx)
}

fn maximum_execute(
    args: &[RuntimeValue],
    _ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let RuntimeValue::Pool(d, _) = &args[0] {
        Ok(numeric_pool(d)
            .sum()
            .ordered_outcomes()
            .last()
            .map(|(o, _)| *o)
            .unwrap_or(0)
            .into())
    } else {
        panic!("wrong argument types to [maximum]");
    }
}

fn choose_execute(
    args: &[RuntimeValue],
    _ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (
        RuntimeValue::Pool(first, first_type),
        RuntimeValue::Element(ElementValue::Int(condition)),
        RuntimeValue::Pool(second, second_type),
    ) = (&args[0], &args[1], &args[2])
    {
        Ok(if *condition == 0 {
            RuntimeValue::Pool(second.clone(), second_type.clone())
        } else {
            RuntimeValue::Pool(first.clone(), first_type.clone())
        })
    } else {
        panic!("wrong argument types to [choose if else]");
    }
}

fn reverse_execute(
    args: &[RuntimeValue],
    _ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    match &args[0] {
        RuntimeValue::List(lst, outcome_type) => Ok(RuntimeValue::List(
            std::rc::Rc::new(lst.iter().rev().cloned().collect()),
            outcome_type.clone(),
        )),
        _ => unreachable!("argument coercion guarantees a sequence"),
    }
}

fn sort_execute(
    args: &[RuntimeValue],
    ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let RuntimeValue::List(lst, outcome_type) = &args[0] {
        let mut lst = (**lst).clone();
        if ctx.lowest_first {
            lst.sort_unstable();
        } else {
            lst.sort_unstable_by_key(|outcome| {
                -outcome
                    .as_int()
                    .expect("sort primitive received a non-numeric sequence")
            });
        }
        Ok(RuntimeValue::List(
            std::rc::Rc::new(lst),
            outcome_type.clone(),
        ))
    } else {
        panic!("wrong argument types to [sort]");
    }
}

/// Builds a tuple from its arguments. Registered for each supported arity; the
/// arity is enforced by function-name matching, not by this executor.
fn tuple_execute(args: &[RuntimeValue], ctx: PrimitiveCtx) -> Result<RuntimeValue, RuntimeError> {
    let invalid = args
        .iter()
        .enumerate()
        .filter(|(_, arg)| matches!(arg, RuntimeValue::Element(ElementValue::Tuple(_))))
        .map(|(index, arg)| argument_error(ctx, index, "a non-tuple value", arg))
        .collect::<Vec<_>>();
    if !invalid.is_empty() {
        return Err(invalid_arguments(
            ctx,
            "cannot contain another tuple",
            "Pass each inner field as its own tuple argument.",
            PrimitiveArgumentErrorKind::Type,
            invalid,
        ));
    }
    let fields = args
        .iter()
        .map(|arg| match arg {
            RuntimeValue::Element(ElementValue::AdditiveIdentity) => ElementValue::Int(0),
            RuntimeValue::Element(value @ (ElementValue::Int(_) | ElementValue::Symbol(_))) => {
                value.clone()
            }
            RuntimeValue::Element(ElementValue::Tuple(_)) => {
                unreachable!("nested tuple arguments were rejected above")
            }
            RuntimeValue::List(_, _) | RuntimeValue::Pool(_, _) => {
                unreachable!("tuple arguments are coerced to elements")
            }
        })
        .collect::<Vec<_>>();
    Ok(RuntimeValue::Element(ElementValue::Tuple(fields.into())))
}

fn field_execute(args: &[RuntimeValue], ctx: PrimitiveCtx) -> Result<RuntimeValue, RuntimeError> {
    const EXPECTED: [&str; 2] = ["an integer", "a tuple"];
    let satisfied = [
        matches!(args[0], RuntimeValue::Element(ElementValue::Int(_))),
        matches!(args[1], RuntimeValue::Element(ElementValue::Tuple(_))),
    ];
    let invalid = satisfied
        .into_iter()
        .enumerate()
        .filter(|(_, satisfied)| !satisfied)
        .map(|(index, _)| argument_error(ctx, index, EXPECTED[index], &args[index]))
        .collect::<Vec<_>>();
    if !invalid.is_empty() {
        return Err(invalid_arguments(
            ctx,
            format!(
                "requires `{}` to be {} and `{}` to be {}",
                argument_name(ctx.identifier, 0),
                EXPECTED[0],
                argument_name(ctx.identifier, 1),
                EXPECTED[1],
            ),
            "Build a tuple with `[tuple A B]`. Tuple field positions start at 1.",
            PrimitiveArgumentErrorKind::Type,
            invalid,
        ));
    }
    let (
        RuntimeValue::Element(ElementValue::Int(index)),
        RuntimeValue::Element(ElementValue::Tuple(fields)),
    ) = (&args[0], &args[1])
    else {
        unreachable!("field argument types were checked above")
    };
    let requested = *index;
    let index = index
        .checked_sub(1)
        .and_then(|index| usize::try_from(index).ok());
    let Some(value) = index.and_then(|index| fields.get(index)) else {
        return Err(RuntimeError::InvalidPrimitiveValue(Box::new(
            PrimitiveValueError {
                range: ctx.function_range.into(),
                function: ctx.identifier,
                requirement: format!("cannot select field `{requested}`"),
                argument: "INDEX",
                found_range: ctx.arg_ranges[0].into(),
                value: args[0].clone(),
                constraint: format!("an index from 1 through {}", fields.len()),
                help: Some("Tuple field positions start at 1.".to_string()),
            },
        )));
    };
    Ok(RuntimeValue::Element(value.clone()))
}

fn is_integer_execute(
    args: &[RuntimeValue],
    _ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    let RuntimeValue::Element(value) = &args[0] else {
        unreachable!("an element argument is enforced by the evaluator")
    };
    // The empty sum has already become `0` on its way to an element argument.
    Ok(i32::from(value.as_int().is_some()).into())
}

/// Shared body of the two `... integers in SEQ` primitives, which differ only
/// in what an `int` outcome contributes.
///
/// Both quantities are additive over the dice, so a pool is handled by mapping
/// each die's outcomes and summing: exact at any dimension, and without
/// enumerating multisets. Outcomes that are not `int`s contribute nothing,
/// which is what lets these take a mixed pool apart.
fn integers_in(arg: &RuntimeValue, of_int: fn(i32) -> i32) -> RuntimeValue {
    let contribution = |value: &ElementValue| match value {
        ElementValue::Int(value) => of_int(*value),
        ElementValue::AdditiveIdentity => of_int(0),
        ElementValue::Symbol(_) | ElementValue::Tuple(_) => 0,
    };
    match arg {
        RuntimeValue::Element(value) => contribution(value).into(),
        RuntimeValue::List(values, _) => values.iter().map(contribution).sum::<i32>().into(),
        RuntimeValue::Pool(pool, _) => {
            let mapped = (**pool)
                .clone()
                .map_outcomes(|outcome| ElementValue::Int(contribution(&outcome)));
            let summed = sum_pool(&mapped, &ElementType::Int).expect("an int pool is additive");
            RuntimeValue::Pool(Rc::new(summed), ElementType::Int)
        }
    }
}

fn sum_integers_execute(
    args: &[RuntimeValue],
    _ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    Ok(integers_in(&args[0], |value| value))
}

fn count_integers_execute(
    args: &[RuntimeValue],
    _ctx: PrimitiveCtx,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    Ok(integers_in(&args[0], |_| 1))
}

fn numeric_pool(pool: &Pool<ElementValue>) -> Pool<i32> {
    pool.clone().map_outcomes(|outcome| {
        outcome
            .as_int()
            .expect("numeric primitive received a non-numeric pool")
    })
}

fn numeric_list(list: &[ElementValue]) -> Vec<i32> {
    list.iter()
        .map(|outcome| {
            outcome
                .as_int()
                .expect("numeric primitive received a non-numeric sequence")
        })
        .collect()
}

macro_rules! define_primitives {
    (
        $(
            $constant:ident:
            $name:literal,
            $arg_types:expr,
            $accepts_non_numeric:literal,
            $execute:path,
            $signature:literal,
            $snippet:literal,
            $documentation:literal,
            $documentation_url:literal;
        )+
    ) => {
        $(
            pub static $constant: Primitive = Primitive {
                identifier: $name,
                arg_types: $arg_types,
                accepts_non_numeric: $accepts_non_numeric,
                execute: $execute,
            };
        )+

        static PRIMITIVES: &[(&str, &Primitive)] = &[
            $(($name, &$constant),)+
        ];

        static PRIMITIVE_METADATA: &[PrimitiveMetadata] = &[
            $(PrimitiveMetadata {
                identifier: $name,
                signature: $signature,
                snippet: $snippet,
                documentation: $documentation,
                documentation_url: $documentation_url,
            },)+
        ];
    };
}

define_primitives! {
    ABSOLUTE_PRIMITIVE:
        "absolute {}", &[Some(StaticType::Int)], false, absolute_execute,
        "[absolute N:n]", "absolute ${N}",
        "Returns the absolute value of N.", "/help/spec/#absolute-nn";
    // Keep SEQ uncoerced so the executor can distinguish a deterministic
    // sequence from a pool and route the latter through Icepool directly.
    CONTAINS_PRIMITIVE:
        "{} contains {}",
        &[None, Some(StaticType::Int)],
        true,
        contains_execute,
        "[SEQ:s contains N:n]", "${SEQ} contains ${N}",
        "Returns 1 when SEQ contains N, or 0 otherwise.", "/help/spec/#seqs-contains-nn";
    // Keep HAYSTACK uncoerced so the executor can distinguish a deterministic
    // sequence from a pool and route the latter through Icepool directly.
    COUNT_PRIMITIVE:
        "count {} in {}",
        &[Some(StaticType::List), None],
        true,
        count_execute,
        "[count NEEDLES:s in HAYSTACK:s]", "count ${NEEDLES} in ${HAYSTACK}",
        "Counts occurrences of every element of NEEDLES in HAYSTACK.", "/help/spec/#count-needless-in-haystacks";
    // Keep SEQ uncoerced for the same reason as `count`: a pool must reach the
    // executor whole, so that it can be mapped and summed rather than
    // enumerated.
    COUNT_INTEGERS_PRIMITIVE:
        "count integers in {}",
        &[None],
        true,
        count_integers_execute,
        "[count integers in SEQ:s]", "count integers in ${SEQ}",
        "Counts the outcomes of SEQ that are integers.", "/help/spec/#-count-integers-in-seqs";
    SUM_INTEGERS_PRIMITIVE:
        "sum integers in {}",
        &[None],
        true,
        sum_integers_execute,
        "[sum integers in SEQ:s]", "sum integers in ${SEQ}",
        "Sums the outcomes of SEQ that are integers, ignoring the rest.", "/help/spec/#-sum-integers-in-seqs";
    IS_INTEGER_PRIMITIVE:
        "{} is integer", &[Some(StaticType::Int)], true, is_integer_execute,
        "[N:n is integer]", "${N} is integer",
        "Returns 1 when N is an integer, or 0 when it is a symbol or a tuple.", "/help/spec/#-nn-is-integer";
    EXPLODE_PRIMITIVE:
        "explode {}", &[Some(StaticType::Pool)], false, explode_execute,
        "[explode POOL:d]", "explode ${POOL}",
        "Rerolls the highest outcome and adds it to the original roll.", "/help/spec/#explode-poold";
    HIGHEST_PRIMITIVE:
        "highest {} of {}",
        &[Some(StaticType::Int), Some(StaticType::Pool)],
        false,
        highest_execute,
        "[highest COUNT:n of POOL:d]", "highest ${COUNT} of ${POOL}",
        "Sums the highest COUNT dice in each outcome of POOL.", "/help/spec/#highest-countn-of-poold-lowest-countn-of-poold-middle-countn-of-poold";
    LOWEST_PRIMITIVE:
        "lowest {} of {}",
        &[Some(StaticType::Int), Some(StaticType::Pool)],
        false,
        lowest_execute,
        "[lowest COUNT:n of POOL:d]", "lowest ${COUNT} of ${POOL}",
        "Sums the lowest COUNT dice in each outcome of POOL.", "/help/spec/#highest-countn-of-poold-lowest-countn-of-poold-middle-countn-of-poold";
    MIDDLE_PRIMITIVE:
        "middle {} of {}",
        &[Some(StaticType::Int), Some(StaticType::Pool)],
        false,
        middle_execute,
        "[middle COUNT:n of POOL:d]", "middle ${COUNT} of ${POOL}",
        "Sums the middle COUNT dice in each outcome of POOL.", "/help/spec/#highest-countn-of-poold-lowest-countn-of-poold-middle-countn-of-poold";
    HIGHEST_OF_PRIMITIVE:
        "highest of {} and {}",
        &[Some(StaticType::Int), Some(StaticType::Int)],
        false,
        highest_of_execute,
        "[highest of FIRST:n and SECOND:n]", "highest of ${FIRST} and ${SECOND}",
        "Returns the greater of FIRST and SECOND.", "/help/spec/#highest-of-firstn-and-secondn-lowest-of-firstn-and-secondn";
    LOWEST_OF_PRIMITIVE:
        "lowest of {} and {}",
        &[Some(StaticType::Int), Some(StaticType::Int)],
        false,
        lowest_of_execute,
        "[lowest of FIRST:n and SECOND:n]", "lowest of ${FIRST} and ${SECOND}",
        "Returns the lesser of FIRST and SECOND.", "/help/spec/#highest-of-firstn-and-secondn-lowest-of-firstn-and-secondn";
    MAXIMUM_PRIMITIVE:
        "maximum of {}", &[Some(StaticType::Pool)], false, maximum_execute,
        "[maximum of POOL:d]", "maximum of ${POOL}",
        "Returns the largest possible outcome of the summed POOL.", "/help/spec/#maximum-of-poold";
    CHOOSE_PRIMITIVE:
        "choose {} if {} else {}",
        &[
            Some(StaticType::Pool),
            Some(StaticType::Int),
            Some(StaticType::Pool),
        ],
        true,
        choose_execute,
        "[choose FIRST:d if CONDITION:n else SECOND:d]", "choose ${FIRST} if ${CONDITION} else ${SECOND}",
        "Returns FIRST when CONDITION is nonzero, and SECOND otherwise.", "/help/spec/#-choose-firstd-if-conditionn-else-secondd";
    REVERSE_PRIMITIVE:
        "reverse {}", &[Some(StaticType::List)], true, reverse_execute,
        "[reverse SEQUENCE:s]", "reverse ${SEQUENCE}",
        "Returns SEQUENCE in reverse order.", "/help/spec/#reverse-sequences";
    SORT_PRIMITIVE:
        "sort {}", &[Some(StaticType::List)], false, sort_execute,
        "[sort SEQUENCE:s]", "sort ${SEQUENCE}",
        "Sorts SEQUENCE according to the position-order setting.", "/help/spec/#sort-sequences";
    EXPLODE_ON_PRIMITIVE:
        "explode {} on {}",
        &[Some(StaticType::Pool), Some(StaticType::List)],
        false,
        explode_on_execute,
        "[explode POOL:d on COND:s]", "explode ${POOL} on ${COND}",
        "Rerolls and adds outcomes of POOL that are contained in COND.", "/help/spec/#explode-poold-on-conds";
    REROLL_PRIMITIVE:
        "reroll {}", &[Some(StaticType::Pool)], false, reroll_execute,
        "[reroll POOL:d]", "reroll ${POOL}",
        "Replaces the highest outcome of POOL with a new roll.", "/help/spec/#-reroll-poold";
    REROLL_ON_PRIMITIVE:
        "reroll {} on {}",
        &[Some(StaticType::Pool), Some(StaticType::List)],
        false,
        reroll_on_execute,
        "[reroll POOL:d on COND:s]", "reroll ${POOL} on ${COND}",
        "Replaces outcomes of POOL that are contained in COND with a new roll.", "/help/spec/#-reroll-poold-on-conds";
    TUPLE_2_PRIMITIVE:
        "tuple {} {}", &[Some(StaticType::Int); 2], true, tuple_execute,
        "[tuple A:n B:n]", "tuple ${A} ${B}",
        "Constructs a two-field tuple.", "/help/spec/#-tuple-an-bn-tuple-an-bn-cn-tuple-an-bn-cn-dn";
    TUPLE_3_PRIMITIVE:
        "tuple {} {} {}", &[Some(StaticType::Int); 3], true, tuple_execute,
        "[tuple A:n B:n C:n]", "tuple ${A} ${B} ${C}",
        "Constructs a three-field tuple.", "/help/spec/#-tuple-an-bn-tuple-an-bn-cn-tuple-an-bn-cn-dn";
    TUPLE_4_PRIMITIVE:
        "tuple {} {} {} {}", &[Some(StaticType::Int); 4], true, tuple_execute,
        "[tuple A:n B:n C:n D:n]", "tuple ${A} ${B} ${C} ${D}",
        "Constructs a four-field tuple.", "/help/spec/#-tuple-an-bn-tuple-an-bn-cn-tuple-an-bn-cn-dn";
    FIELD_PRIMITIVE:
        "field {} of {}", &[Some(StaticType::Int); 2], true, field_execute,
        "[field INDEX:n of TUPLE:n]", "field ${INDEX} of ${TUPLE}",
        "Returns the one-based INDEX field of TUPLE.", "/help/spec/#-field-indexn-of-tuplen";
}

fn keep_list_for_primitive(
    mode: KeepMode,
    identifier: &'static str,
    keep: i32,
    keep_range: ast::Range,
    pool_size: usize,
    function_range: ast::Range,
) -> Result<Vec<bool>, crate::eval::RuntimeError> {
    use crate::eval::RuntimeError;

    if keep < 0 {
        return Err(RuntimeError::NegativeArgumentToFunction {
            range: function_range.into(),
            name: identifier.to_string(),
            found_range: keep_range.into(),
            value: keep,
        });
    }
    let keep = usize::try_from(keep).expect("keep is positive");
    if keep >= pool_size {
        return Ok(vec![true; pool_size]);
    }
    let mut keep_list = vec![false; pool_size];
    if keep == 0 {
        return Ok(keep_list);
    }
    match mode {
        KeepMode::Highest => {
            keep_list[pool_size - keep..].fill(true);
        }
        KeepMode::Lowest => {
            keep_list[..keep].fill(true);
        }
        KeepMode::Middle => {
            // This is rounding down
            let start = (pool_size - keep).div_ceil(2);
            keep_list[start..start + keep].fill(true);
        }
    }
    Ok(keep_list)
}

pub fn register_primitives(functions: &mut HashMap<String, Function>) {
    functions.extend(
        PRIMITIVES
            .iter()
            .map(|(name, primitive)| (name.to_string(), Function::Primitive(primitive))),
    );
}

/// Returns completion and documentation metadata for every built-in function.
pub fn primitive_metadata() -> &'static [PrimitiveMetadata] {
    PRIMITIVE_METADATA
}

pub(crate) fn primitive_signature(identifier: &str) -> Option<&'static str> {
    PRIMITIVE_METADATA
        .iter()
        .find(|primitive| primitive.identifier == identifier)
        .map(|primitive| primitive.signature)
}

#[cfg(test)]
mod tests {
    use malachite::Natural;

    use super::*;
    use crate::{ast::Range, dice::Pool};

    fn dummy_range() -> Range {
        Range { start: 0, end: 0 }
    }

    fn ctx(arg_ranges: &[Range], explode_depth: usize, lowest_first: bool) -> PrimitiveCtx<'_> {
        PrimitiveCtx {
            arg_ranges,
            explode_depth,
            lowest_first,
            function_range: dummy_range(),
            identifier: "test {}",
        }
    }

    fn int_value(value: i32) -> RuntimeValue {
        value.into()
    }

    fn list_value(values: Vec<i32>) -> RuntimeValue {
        values.into()
    }

    fn pool_value(value: Pool) -> RuntimeValue {
        value.into()
    }

    fn to_nat_list(outcomes: &[(ElementValue, Natural)]) -> Vec<(i32, i32)> {
        outcomes
            .iter()
            .map(|(i, w)| (i.as_int().unwrap(), w.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn test_contains_execute() {
        let haystack = vec![1, 2, 3, 3, 4, 5];
        let args = vec![list_value(haystack), int_value(3)];
        let result = contains_execute(&args, ctx(&[], 0, false)).unwrap();
        assert_eq!(result, int_value(1));

        let haystack = vec![1, 2, 4, 5];
        let args = vec![list_value(haystack), int_value(3)];
        let result = contains_execute(&args, ctx(&[], 0, false)).unwrap();
        assert_eq!(result, int_value(0));

        let args = vec![int_value(3), int_value(3)];
        assert_eq!(
            contains_execute(&args, ctx(&[], 0, false)).unwrap(),
            int_value(1)
        );

        let args = vec![pool_value(Pool::ndn(3, 6)), int_value(6)];
        let result = contains_execute(&args, ctx(&[], 0, false)).unwrap();
        let RuntimeValue::Pool(result, _) = result else {
            panic!("Expected pool result");
        };
        assert_eq!(to_nat_list(result.ordered_outcomes()), [(0, 125), (1, 91)]);
    }

    #[test]
    fn test_count_execute() {
        let needle = vec![1, 2, 2];
        let haystack = vec![1, 2, 2, 3, 2, 1];
        let args = vec![list_value(needle), list_value(haystack)];
        let result = count_execute(&args, ctx(&[], 0, false)).unwrap();
        // Should count: 1 appears 2 times, 2 appears 3 times -> 1*2 + 2*3 = 8
        assert_eq!(result, int_value(8));

        let needles = vec![2, 4, 6];
        let haystack = Pool::from_list(3, (1..=6).collect());
        let args = vec![list_value(needles), pool_value(haystack)];
        let result = count_execute(&args, ctx(&[], 0, false)).unwrap();
        let RuntimeValue::Pool(result, _) = result else {
            panic!("Expected pool result");
        };
        assert_eq!(
            to_nat_list(result.ordered_outcomes()),
            [(0, 27), (1, 81), (2, 81), (3, 27)]
        );

        let args = vec![list_value(vec![1, 2, 2]), int_value(2)];
        assert_eq!(
            count_execute(&args, ctx(&[], 0, false)).unwrap(),
            int_value(2)
        );
    }

    #[test]
    fn test_explode_execute() {
        let pool = Pool::from_list(1, vec![1, 2, 3]);
        let args = vec![pool_value(pool)];
        let result = explode_execute(&args, ctx(&[], 2, false)).unwrap();
        // Should return a pool where 3s explode
        if let RuntimeValue::Pool(result_pool, _) = result {
            assert_eq!(
                to_nat_list(result_pool.ordered_outcomes()),
                [(1, 9), (2, 9), (4, 3), (5, 3), (7, 1), (8, 1), (9, 1)]
            );
        } else {
            panic!("Expected pool result");
        }
    }

    #[test]
    fn test_highest_execute() {
        let pool = Pool::from_list(3, vec![1, 2, 3, 4, 5, 6]);
        let args = vec![int_value(2), pool_value(pool)];
        let ranges = vec![dummy_range(), dummy_range()];
        let result = highest_execute(&args, ctx(&ranges, 0, false)).unwrap();
        // Should keep the highest 2 dice from a 3d6 roll
        if let RuntimeValue::Pool(result_pool, _) = result {
            assert_eq!(
                to_nat_list(result_pool.ordered_outcomes()),
                [
                    (2, 1),
                    (3, 3),
                    (4, 7),
                    (5, 12),
                    (6, 19),
                    (7, 27),
                    (8, 34),
                    (9, 36),
                    (10, 34),
                    (11, 27),
                    (12, 16)
                ]
            );
        } else {
            panic!("Expected pool result");
        }
    }

    #[test]
    fn test_lowest_execute() {
        let pool = Pool::from_list(3, vec![1, 2, 3, 4, 5, 6]);
        let args = vec![int_value(2), pool_value(pool)];
        let ranges = vec![dummy_range(), dummy_range()];
        let result = lowest_execute(&args, ctx(&ranges, 0, false)).unwrap();
        // Should keep the lowest 2 dice from a 3d6 roll
        if let RuntimeValue::Pool(result_pool, _) = result {
            assert_eq!(
                to_nat_list(result_pool.ordered_outcomes()),
                [
                    (2, 16),
                    (3, 27),
                    (4, 34),
                    (5, 36),
                    (6, 34),
                    (7, 27),
                    (8, 19),
                    (9, 12),
                    (10, 7),
                    (11, 3),
                    (12, 1)
                ]
            );
        } else {
            panic!("Expected pool result");
        }
    }

    #[test]
    fn test_middle_execute() {
        let pool = Pool::from_list(3, vec![1, 2, 3, 4, 5, 6]);
        let args = vec![int_value(1), pool_value(pool)];
        let ranges = vec![dummy_range(), dummy_range()];
        let result = middle_execute(&args, ctx(&ranges, 0, false)).unwrap();
        // Should keep the middle 1 die from a 3d6 roll
        if let RuntimeValue::Pool(result_pool, _) = result {
            assert_eq!(
                to_nat_list(result_pool.ordered_outcomes()),
                [(1, 16), (2, 40), (3, 52), (4, 52), (5, 40), (6, 16)]
            );
        } else {
            panic!("Expected pool result");
        }
    }

    #[test]
    fn test_maximum_execute() {
        let pool = Pool::from_list(1, vec![1, 3, 2]);
        let args = vec![pool_value(pool)];
        let result = maximum_execute(&args, ctx(&[], 0, false)).unwrap();
        assert_eq!(result, int_value(3));

        // Test empty pool
        let empty_pool = Pool::from_list(1, vec![]);
        let args = vec![pool_value(empty_pool)];
        let result = maximum_execute(&args, ctx(&[], 0, false)).unwrap();
        assert_eq!(result, int_value(0));
    }

    #[test]
    fn test_choose_execute() {
        let first = pool_value(Pool::ndn(2, 6));
        let second = pool_value(Pool::ndn(1, 20));

        let result = choose_execute(
            &[first.clone(), int_value(-1), second.clone()],
            ctx(&[], 0, false),
        )
        .unwrap();
        assert_eq!(result, first);

        let result =
            choose_execute(&[first, int_value(0), second.clone()], ctx(&[], 0, false)).unwrap();
        assert_eq!(result, second);
    }

    #[test]
    fn test_choose_with_distribution_condition() {
        let statements = crate::grammar::BodyParser::new()
            .parse("output [choose d2 if d{0, 1:3} else d{10, 20}]")
            .unwrap();
        let mut evaluator = crate::eval::Evaluator::new();
        for statement in statements {
            evaluator.execute(&statement).unwrap();
        }

        let outputs = evaluator.take_outputs();
        let [output] = outputs.as_slice() else {
            panic!("Expected one pool output");
        };
        let RuntimeValue::Pool(result, _) = &output.value else {
            panic!("Expected one pool output");
        };
        assert_eq!(
            to_nat_list(result.ordered_outcomes()),
            [(1, 3), (2, 3), (10, 1), (20, 1)]
        );
    }

    #[test]
    fn test_reverse_execute() {
        let list = vec![1, 2, 3, 4, 5];
        let args = vec![list_value(list)];
        let result = reverse_execute(&args, ctx(&[], 0, false)).unwrap();
        if let RuntimeValue::List(reversed, _) = result {
            assert_eq!(numeric_list(&reversed), vec![5, 4, 3, 2, 1]);
        } else {
            panic!("Expected list result");
        }
    }

    #[test]
    fn test_sort_execute() {
        let list = vec![3, 1, 4, 1, 5];
        let args = vec![list_value(list)];

        // Test lowest first (ascending)
        let result = sort_execute(&args, ctx(&[], 0, true)).unwrap();
        if let RuntimeValue::List(sorted, _) = result {
            assert_eq!(numeric_list(&sorted), vec![1, 1, 3, 4, 5]);
        } else {
            panic!("Expected list result");
        }

        // Test highest first (descending)
        let list = vec![3, 1, 4, 1, 5];
        let args = vec![list_value(list)];
        let result = sort_execute(&args, ctx(&[], 0, false)).unwrap();
        if let RuntimeValue::List(sorted, _) = result {
            assert_eq!(numeric_list(&sorted), vec![5, 4, 3, 1, 1]);
        } else {
            panic!("Expected list result");
        }
    }

    #[test]
    fn test_explode_on_execute() {
        let pool = Pool::from_list(1, vec![1, 2, 3]);
        let condition = vec![2, 3];
        let args = vec![pool_value(pool), list_value(condition)];
        let result = explode_on_execute(&args, ctx(&[], 2, false)).unwrap();
        // Should return a pool where 2s and 3s explode
        if let RuntimeValue::Pool(result_pool, _) = result {
            // 1 stays as 1, 2 explodes to 2+reroll, 3 explodes to 3+reroll
            assert_eq!(
                to_nat_list(result_pool.ordered_outcomes()),
                [
                    (1, 9),
                    (3, 3),
                    (4, 3),
                    (5, 1),
                    (6, 3),
                    (7, 4),
                    (8, 3),
                    (9, 1)
                ]
            );
        } else {
            panic!("Expected pool result");
        }
    }

    #[test]
    fn test_reroll_execute() {
        let pool = Pool::from_list(1, vec![1, 2, 3]);
        let args = vec![pool_value(pool)];
        let result = reroll_execute(&args, ctx(&[], 2, false)).unwrap();
        // Should return a pool where 3s reroll (without keeping the original value)
        if let RuntimeValue::Pool(result_pool, _) = result {
            assert_eq!(
                to_nat_list(result_pool.ordered_outcomes()),
                [(1, 13), (2, 13), (3, 1)]
            );
        } else {
            panic!("Expected pool result");
        }
    }

    #[test]
    fn test_reroll_on_execute() {
        let pool = Pool::from_list(1, vec![1, 2, 3, 4]);
        let condition = vec![1, 4];
        let args = vec![pool_value(pool), list_value(condition)];
        let result = reroll_on_execute(&args, ctx(&[], 2, false)).unwrap();
        // Should return a pool where 1s and 4s reroll
        if let RuntimeValue::Pool(result_pool, _) = result {
            assert_eq!(
                to_nat_list(result_pool.ordered_outcomes()),
                [(1, 4), (2, 28), (3, 28), (4, 4)]
            );
        } else {
            panic!("Expected pool result");
        }
    }

    #[test]
    fn primitive_metadata_matches_the_runtime_registry() {
        assert_eq!(PRIMITIVE_METADATA.len(), PRIMITIVES.len());

        for ((identifier, primitive), metadata) in PRIMITIVES.iter().zip(PRIMITIVE_METADATA) {
            assert_eq!(metadata.identifier, *identifier);
            assert_eq!(metadata.identifier, primitive.identifier);
            assert_eq!(
                metadata.snippet.matches("${").count(),
                primitive.arg_types.len()
            );
            // `argument_name` indexes into these, so every argument needs one.
            assert_eq!(
                argument_names(identifier).count(),
                primitive.arg_types.len(),
                "{identifier}"
            );
            assert!(metadata.signature.starts_with('['));
            assert!(metadata.signature.ends_with(']'));
            assert!(!metadata.documentation.is_empty());
            assert!(metadata.documentation_url.starts_with("/help/spec/#"));
        }
    }
}
