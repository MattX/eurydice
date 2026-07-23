use std::collections::HashMap;

use lazy_static::lazy_static;

use crate::{
    ast::{self, StaticType},
    dice::{explode, reroll, Pool},
    eval::{ElementValue, Function, RuntimeError, RuntimeValue},
};

type PrimitiveExecutor = fn(
    &[RuntimeValue],
    &[ast::Range],
    usize,      // explode_depth
    bool,       // lowest_first
    ast::Range, // function_range
) -> Result<RuntimeValue, crate::eval::RuntimeError>;

#[derive(Debug)]
pub struct Primitive {
    pub arg_types: Vec<Option<StaticType>>,
    pub accepts_non_numeric: bool,
    pub execute: PrimitiveExecutor,
}

#[derive(Debug, Clone, Copy)]
enum PrimitiveType {
    Highest,
    Lowest,
    Middle,
}

fn absolute_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    let arg = &args[0];
    Ok(arg.map_numeric_outcomes(i32::abs))
}

fn contains_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    let error = || crate::eval::RuntimeError::EnumTypeError {
        range: function_range.into(),
        message: "[contains] requires a sequence and element with the same outcome type".to_string(),
    };
    let (RuntimeValue::List(_, _), RuntimeValue::Element(_)) = (&args[0], &args[1]) else {
        return Err(error());
    };
    let outcome_type = args[0]
        .merged_outcome_type(&args[1])
        .ok_or_else(error)?
        .summed_type();
    let haystack = args[0].materialize_identities(&outcome_type);
    let needle = args[1].materialize_identities(&outcome_type);
    let (RuntimeValue::List(haystack, _), RuntimeValue::Element(needle)) = (haystack, needle)
    else {
        unreachable!("contains argument shapes were checked")
    };
    let result = haystack.iter().any(|value| value == &needle);
    Ok(if result { 1.into() } else { 0.into() })
}

fn count_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    let error = || crate::eval::RuntimeError::EnumTypeError {
        range: function_range.into(),
        message: "[count] requires sequences with the same outcome type".to_string(),
    };
    if !matches!(
        (&args[0], &args[1]),
        (RuntimeValue::List(_, _), RuntimeValue::List(_, _))
    ) {
        return Err(error());
    }
    let outcome_type = args[0]
        .merged_outcome_type(&args[1])
        .ok_or_else(error)?
        .summed_type();
    let needle = args[0].materialize_identities(&outcome_type);
    let haystack = args[1].materialize_identities(&outcome_type);
    let (RuntimeValue::List(needle, _), RuntimeValue::List(haystack, _)) = (needle, haystack)
    else {
        unreachable!("count argument shapes were checked")
    };
    let (needle, haystack) = (needle.as_slice(), haystack.as_slice());
    let mut needle_map = HashMap::new();
    for n in needle.iter() {
        *needle_map.entry(n).or_insert(0) += 1;
    }
    Ok(haystack
        .iter()
        .map(|item| needle_map.get(item).copied().unwrap_or(0))
        .sum::<i32>()
        .into())
}

fn explode_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let RuntimeValue::Pool(d, _) = &args[0] {
        if d.is_empty() {
            return Ok(args[0].clone());
        }
        let die: Vec<_> = numeric_pool(d).sum().into_die_iter().collect();
        let highest_value = die.last().unwrap().0;
        Ok(Pool::from(explode(die, &[highest_value], explode_depth)).into())
    } else {
        panic!("wrong argument types to [explode]");
    }
}

fn explode_on_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (RuntimeValue::Pool(d, _), RuntimeValue::List(cond, _)) = (&args[0], &args[1]) {
        if d.is_empty() {
            return Ok(args[0].clone());
        }
        let die: Vec<_> = numeric_pool(d).sum().into_die_iter().collect();
        Ok(Pool::from(explode(die, &numeric_list(cond), explode_depth)).into())
    } else {
        panic!("wrong argument types to [explode on]");
    }
}

fn reroll_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let RuntimeValue::Pool(d, _) = &args[0] {
        if d.is_empty() {
            return Ok(args[0].clone());
        }
        let die: Vec<_> = numeric_pool(d).sum().into_die_iter().collect();
        let highest_value = die.last().unwrap().0;
        Ok(Pool::from(reroll(die, &[highest_value], explode_depth)).into())
    } else {
        panic!("wrong argument types to [reroll]");
    }
}

fn reroll_on_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (RuntimeValue::Pool(d, _), RuntimeValue::List(cond, _)) = (&args[0], &args[1]) {
        if d.is_empty() {
            return Ok(args[0].clone());
        }
        let die: Vec<_> = numeric_pool(d).sum().into_die_iter().collect();
        Ok(Pool::from(reroll(die, &numeric_list(cond), explode_depth)).into())
    } else {
        panic!("wrong argument types to [reroll on]");
    }
}

fn highest_execute(
    args: &[RuntimeValue],
    arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (RuntimeValue::Element(ElementValue::Int(i)), RuntimeValue::Pool(d, _)) =
        (&args[0], &args[1])
    {
        let keep_list = keep_list_for_primitive(
            PrimitiveType::Highest,
            *i,
            arg_ranges[0],
            d.dimension() as usize,
            function_range,
        )?;
        Ok(numeric_pool(d).sum_with_keep_list(&keep_list).into())
    } else {
        panic!("wrong argument types to [highest]");
    }
}

fn lowest_execute(
    args: &[RuntimeValue],
    arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (RuntimeValue::Element(ElementValue::Int(i)), RuntimeValue::Pool(d, _)) =
        (&args[0], &args[1])
    {
        let keep_list = keep_list_for_primitive(
            PrimitiveType::Lowest,
            *i,
            arg_ranges[0],
            d.dimension() as usize,
            function_range,
        )?;
        Ok(numeric_pool(d).sum_with_keep_list(&keep_list).into())
    } else {
        panic!("wrong argument types to [lowest]");
    }
}

fn middle_execute(
    args: &[RuntimeValue],
    arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (RuntimeValue::Element(ElementValue::Int(i)), RuntimeValue::Pool(d, _)) =
        (&args[0], &args[1])
    {
        let keep_list = keep_list_for_primitive(
            PrimitiveType::Middle,
            *i,
            arg_ranges[0],
            d.dimension() as usize,
            function_range,
        )?;
        Ok(numeric_pool(d).sum_with_keep_list(&keep_list).into())
    } else {
        panic!("wrong argument types to [middle]");
    }
}

fn highest_of_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (
        RuntimeValue::Element(ElementValue::Int(i)),
        RuntimeValue::Element(ElementValue::Int(j)),
    ) = (&args[0], &args[1])
    {
        Ok((*(i.max(j))).into())
    } else {
        panic!("wrong argument types to [highest of]");
    }
}

fn lowest_of_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (
        RuntimeValue::Element(ElementValue::Int(i)),
        RuntimeValue::Element(ElementValue::Int(j)),
    ) = (&args[0], &args[1])
    {
        Ok((*(i.min(j))).into())
    } else {
        panic!("wrong argument types to [lowest of]");
    }
}

fn maximum_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
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
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
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
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
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
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let RuntimeValue::List(lst, outcome_type) = &args[0] {
        let mut lst = (**lst).clone();
        if lowest_first {
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

fn tuple_execute(
    args: &[RuntimeValue],
    function_range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    let fields = args
        .iter()
        .map(|arg| match arg {
            RuntimeValue::Element(ElementValue::AdditiveIdentity) => Ok(ElementValue::Int(0)),
            RuntimeValue::Element(value @ (ElementValue::Int(_) | ElementValue::Enum { .. })) => {
                Ok(value.clone())
            }
            RuntimeValue::Element(ElementValue::Tuple(_)) => Err(RuntimeError::EnumTypeError {
                range: function_range.into(),
                message: "nested tuples are not supported".to_string(),
            }),
            RuntimeValue::List(_, _) | RuntimeValue::Pool(_, _) => {
                unreachable!("tuple arguments are coerced to elements")
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RuntimeValue::Element(ElementValue::Tuple(fields.into())))
}

macro_rules! tuple_executor {
    ($name:ident) => {
        fn $name(
            args: &[RuntimeValue],
            _arg_ranges: &[ast::Range],
            _explode_depth: usize,
            _lowest_first: bool,
            function_range: ast::Range,
        ) -> Result<RuntimeValue, RuntimeError> {
            tuple_execute(args, function_range)
        }
    };
}

tuple_executor!(tuple_2_execute);
tuple_executor!(tuple_3_execute);
tuple_executor!(tuple_4_execute);

fn element_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    function_range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    let (
        RuntimeValue::Element(ElementValue::Int(index)),
        RuntimeValue::Element(ElementValue::Tuple(fields)),
    ) = (&args[0], &args[1])
    else {
        return Err(RuntimeError::EnumTypeError {
            range: function_range.into(),
            message: "[element I of T] requires an integer index and a tuple".to_string(),
        });
    };
    let index = index
        .checked_sub(1)
        .and_then(|index| usize::try_from(index).ok());
    let Some(value) = index.and_then(|index| fields.get(index)) else {
        return Err(RuntimeError::EnumTypeError {
            range: function_range.into(),
            message: format!("tuple index must be between 1 and {}", fields.len()),
        });
    };
    Ok(RuntimeValue::Element(value.clone()))
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

lazy_static! {
    pub static ref ABSOLUTE_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int)],
        accepts_non_numeric: false,
        execute: absolute_execute,
    };
    pub static ref CONTAINS_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::List), Some(StaticType::Int)],
        accepts_non_numeric: true,
        execute: contains_execute,
    };
    pub static ref COUNT_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::List), Some(StaticType::List)],
        accepts_non_numeric: true,
        execute: count_execute,
    };
    pub static ref EXPLODE_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Pool)],
        accepts_non_numeric: false,
        execute: explode_execute,
    };
    pub static ref HIGHEST_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int), Some(StaticType::Pool)],
        accepts_non_numeric: false,
        execute: highest_execute,
    };
    pub static ref LOWEST_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int), Some(StaticType::Pool)],
        accepts_non_numeric: false,
        execute: lowest_execute,
    };
    pub static ref MIDDLE_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int), Some(StaticType::Pool)],
        accepts_non_numeric: false,
        execute: middle_execute,
    };
    pub static ref HIGHEST_OF_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int), Some(StaticType::Int)],
        accepts_non_numeric: false,
        execute: highest_of_execute,
    };
    pub static ref LOWEST_OF_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int), Some(StaticType::Int)],
        accepts_non_numeric: false,
        execute: lowest_of_execute,
    };
    pub static ref MAXIMUM_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Pool)],
        accepts_non_numeric: false,
        execute: maximum_execute,
    };
    pub static ref CHOOSE_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![
            Some(StaticType::Pool),
            Some(StaticType::Int),
            Some(StaticType::Pool),
        ],
        accepts_non_numeric: true,
        execute: choose_execute,
    };
    pub static ref REVERSE_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::List)],
        accepts_non_numeric: true,
        execute: reverse_execute,
    };
    pub static ref SORT_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::List)],
        accepts_non_numeric: false,
        execute: sort_execute,
    };
    pub static ref EXPLODE_ON_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Pool), Some(StaticType::List)],
        accepts_non_numeric: false,
        execute: explode_on_execute,
    };
    pub static ref REROLL_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Pool)],
        accepts_non_numeric: false,
        execute: reroll_execute,
    };
    pub static ref REROLL_ON_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Pool), Some(StaticType::List)],
        accepts_non_numeric: false,
        execute: reroll_on_execute,
    };
    pub static ref TUPLE_2_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int); 2],
        accepts_non_numeric: true,
        execute: tuple_2_execute,
    };
    pub static ref TUPLE_3_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int); 3],
        accepts_non_numeric: true,
        execute: tuple_3_execute,
    };
    pub static ref TUPLE_4_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int); 4],
        accepts_non_numeric: true,
        execute: tuple_4_execute,
    };
    pub static ref ELEMENT_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int); 2],
        accepts_non_numeric: true,
        execute: element_execute,
    };
}

fn keep_list_for_primitive(
    primitive: PrimitiveType,
    keep: i32,
    keep_range: ast::Range,
    outcomes_size: usize,
    function_range: ast::Range,
) -> Result<Vec<bool>, crate::eval::RuntimeError> {
    use crate::eval::RuntimeError;

    if keep < 0 {
        return Err(RuntimeError::NegativeArgumentToFunction {
            range: function_range.into(),
            name: match primitive {
                PrimitiveType::Highest => "highest".to_string(),
                PrimitiveType::Lowest => "lowest".to_string(),
                PrimitiveType::Middle => "middle".to_string(),
            },
            found_range: keep_range.into(),
            value: keep,
        });
    }
    let keep = usize::try_from(keep).expect("keep is positive");
    if keep >= outcomes_size {
        return Ok(vec![true; outcomes_size]);
    }
    let mut keep_list = vec![false; outcomes_size];
    if keep == 0 {
        return Ok(keep_list);
    }
    match primitive {
        PrimitiveType::Highest => {
            keep_list[outcomes_size - keep..].fill(true);
        }
        PrimitiveType::Lowest => {
            keep_list[..keep].fill(true);
        }
        PrimitiveType::Middle => {
            // This is rounding down
            let start = (outcomes_size - keep).div_ceil(2);
            keep_list[start..start + keep].fill(true);
        }
    }
    Ok(keep_list)
}

pub fn register_primitives(functions: &mut HashMap<String, Function>) {
    functions.insert(
        "absolute {}".to_string(),
        Function::Primitive(&ABSOLUTE_PRIMITIVE),
    );
    functions.insert(
        "{} contains {}".to_string(),
        Function::Primitive(&CONTAINS_PRIMITIVE),
    );
    functions.insert(
        "count {} in {}".to_string(),
        Function::Primitive(&COUNT_PRIMITIVE),
    );
    functions.insert(
        "explode {}".to_string(),
        Function::Primitive(&EXPLODE_PRIMITIVE),
    );
    functions.insert(
        "highest {} of {}".to_string(),
        Function::Primitive(&HIGHEST_PRIMITIVE),
    );
    functions.insert(
        "lowest {} of {}".to_string(),
        Function::Primitive(&LOWEST_PRIMITIVE),
    );
    functions.insert(
        "middle {} of {}".to_string(),
        Function::Primitive(&MIDDLE_PRIMITIVE),
    );
    functions.insert(
        "highest of {} and {}".to_string(),
        Function::Primitive(&HIGHEST_OF_PRIMITIVE),
    );
    functions.insert(
        "lowest of {} and {}".to_string(),
        Function::Primitive(&LOWEST_OF_PRIMITIVE),
    );
    functions.insert(
        "maximum of {}".to_string(),
        Function::Primitive(&MAXIMUM_PRIMITIVE),
    );
    functions.insert(
        "choose {} if {} else {}".to_string(),
        Function::Primitive(&CHOOSE_PRIMITIVE),
    );
    functions.insert(
        "reverse {}".to_string(),
        Function::Primitive(&REVERSE_PRIMITIVE),
    );
    functions.insert("sort {}".to_string(), Function::Primitive(&SORT_PRIMITIVE));
    functions.insert(
        "explode {} on {}".to_string(),
        Function::Primitive(&EXPLODE_ON_PRIMITIVE),
    );
    functions.insert(
        "reroll {}".to_string(),
        Function::Primitive(&REROLL_PRIMITIVE),
    );
    functions.insert(
        "reroll {} on {}".to_string(),
        Function::Primitive(&REROLL_ON_PRIMITIVE),
    );
    functions.insert(
        "tuple {} {}".to_string(),
        Function::Primitive(&TUPLE_2_PRIMITIVE),
    );
    functions.insert(
        "tuple {} {} {}".to_string(),
        Function::Primitive(&TUPLE_3_PRIMITIVE),
    );
    functions.insert(
        "tuple {} {} {} {}".to_string(),
        Function::Primitive(&TUPLE_4_PRIMITIVE),
    );
    functions.insert(
        "element {} of {}".to_string(),
        Function::Primitive(&ELEMENT_PRIMITIVE),
    );
}

#[cfg(test)]
mod tests {
    use malachite::Natural;

    use super::*;
    use crate::{ast::Range, dice::Pool};

    fn dummy_range() -> Range {
        Range { start: 0, end: 0 }
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
    fn test_absolute_execute() {
        let args = vec![int_value(-5)];
        let result = absolute_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, int_value(5));

        let args = vec![int_value(3)];
        let result = absolute_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, int_value(3));
    }

    #[test]
    fn test_contains_execute() {
        let haystack = vec![1, 2, 3, 3, 4, 5];
        let args = vec![list_value(haystack), int_value(3)];
        let result = contains_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, int_value(1));

        let haystack = vec![1, 2, 4, 5];
        let args = vec![list_value(haystack), int_value(3)];
        let result = contains_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, int_value(0));
    }

    #[test]
    fn test_count_execute() {
        let needle = vec![1, 2, 2];
        let haystack = vec![1, 2, 2, 3, 2, 1];
        let args = vec![list_value(needle), list_value(haystack)];
        let result = count_execute(&args, &[], 0, false, dummy_range()).unwrap();
        // Should count: 1 appears 2 times, 2 appears 3 times -> 1*2 + 2*3 = 8
        assert_eq!(result, int_value(8));
    }

    #[test]
    fn test_explode_execute() {
        let pool = Pool::from_list(1, vec![1, 2, 3]);
        let args = vec![pool_value(pool)];
        let result = explode_execute(&args, &[], 2, false, dummy_range()).unwrap();
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
        let result = highest_execute(&args, &ranges, 0, false, dummy_range()).unwrap();
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
        let result = lowest_execute(&args, &ranges, 0, false, dummy_range()).unwrap();
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
        let result = middle_execute(&args, &ranges, 0, false, dummy_range()).unwrap();
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
    fn test_highest_of_execute() {
        let args = vec![int_value(5), int_value(3)];
        let result = highest_of_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, int_value(5));

        let args = vec![int_value(2), int_value(7)];
        let result = highest_of_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, int_value(7));
    }

    #[test]
    fn test_lowest_of_execute() {
        let args = vec![int_value(5), int_value(3)];
        let result = lowest_of_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, int_value(3));

        let args = vec![int_value(2), int_value(7)];
        let result = lowest_of_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, int_value(2));
    }

    #[test]
    fn test_maximum_execute() {
        let pool = Pool::from_list(1, vec![1, 3, 2]);
        let args = vec![pool_value(pool)];
        let result = maximum_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, int_value(3));

        // Test empty pool
        let empty_pool = Pool::from_list(1, vec![]);
        let args = vec![pool_value(empty_pool)];
        let result = maximum_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, int_value(0));
    }

    #[test]
    fn test_choose_execute() {
        let first = pool_value(Pool::ndn(2, 6));
        let second = pool_value(Pool::ndn(1, 20));

        let result = choose_execute(
            &[first.clone(), int_value(-1), second.clone()],
            &[],
            0,
            false,
            dummy_range(),
        )
        .unwrap();
        assert_eq!(result, first);

        let result = choose_execute(
            &[first, int_value(0), second.clone()],
            &[],
            0,
            false,
            dummy_range(),
        )
        .unwrap();
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
        let result = reverse_execute(&args, &[], 0, false, dummy_range()).unwrap();
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
        let result = sort_execute(&args, &[], 0, true, dummy_range()).unwrap();
        if let RuntimeValue::List(sorted, _) = result {
            assert_eq!(numeric_list(&sorted), vec![1, 1, 3, 4, 5]);
        } else {
            panic!("Expected list result");
        }

        // Test highest first (descending)
        let list = vec![3, 1, 4, 1, 5];
        let args = vec![list_value(list)];
        let result = sort_execute(&args, &[], 0, false, dummy_range()).unwrap();
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
        let result = explode_on_execute(&args, &[], 2, false, dummy_range()).unwrap();
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
        let result = reroll_execute(&args, &[], 2, false, dummy_range()).unwrap();
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
        let result = reroll_on_execute(&args, &[], 2, false, dummy_range()).unwrap();
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
    fn test_register_primitives() {
        use std::collections::HashMap;
        let mut functions = HashMap::new();
        register_primitives(&mut functions);

        // Check that all expected primitives are registered
        assert!(functions.contains_key("absolute {}"));
        assert!(functions.contains_key("{} contains {}"));
        assert!(functions.contains_key("count {} in {}"));
        assert!(functions.contains_key("explode {}"));
        assert!(functions.contains_key("explode {} on {}"));
        assert!(functions.contains_key("reroll {}"));
        assert!(functions.contains_key("reroll {} on {}"));
        assert!(functions.contains_key("highest {} of {}"));
        assert!(functions.contains_key("lowest {} of {}"));
        assert!(functions.contains_key("middle {} of {}"));
        assert!(functions.contains_key("highest of {} and {}"));
        assert!(functions.contains_key("lowest of {} and {}"));
        assert!(functions.contains_key("maximum of {}"));
        assert!(functions.contains_key("choose {} if {} else {}"));
        assert!(functions.contains_key("reverse {}"));
        assert!(functions.contains_key("sort {}"));
        assert!(functions.contains_key("tuple {} {}"));
        assert!(functions.contains_key("tuple {} {} {}"));
        assert!(functions.contains_key("tuple {} {} {} {}"));
        assert!(functions.contains_key("element {} of {}"));

        // Should have registered exactly 20 functions
        assert_eq!(functions.len(), 20);
    }
}
