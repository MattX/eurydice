use std::collections::HashMap;

use lazy_static::lazy_static;

use crate::{
    ast::{self, StaticType},
    dice::{explode, reroll, Pool},
    eval::{Function, RuntimeValue},
};

#[derive(Debug)]
pub struct Primitive {
    pub arg_types: Vec<Option<StaticType>>,
    pub execute: fn(
        &[RuntimeValue],
        &[ast::Range],
        usize,      // explode_depth
        bool,       // lowest_first
        ast::Range, // function_range
    ) -> Result<RuntimeValue, crate::eval::RuntimeError>,
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
    Ok(arg.map_outcomes(|o| o.abs()))
}

fn contains_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (RuntimeValue::List(haystack), RuntimeValue::Int(needle)) = (&args[0], &args[1]) {
        Ok(if haystack.iter().any(|h| h == needle) {
            1.into()
        } else {
            0.into()
        })
    } else {
        panic!("wrong argument types to [contains]");
    }
}

fn count_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let (RuntimeValue::List(needle), RuntimeValue::List(haystack)) = (&args[0], &args[1]) {
        let mut needle_map = HashMap::new();
        for n in needle.iter() {
            *needle_map.entry(n).or_insert(0) += 1;
        }
        Ok(haystack
            .iter()
            .map(|item| needle_map.get(item).copied().unwrap_or(0))
            .sum::<i32>()
            .into())
    } else {
        panic!("wrong argument types to [count]");
    }
}

fn explode_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    explode_depth: usize,
    _lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let RuntimeValue::Pool(d) = &args[0] {
        if d.is_empty() {
            return Ok(Pool::from_list(1, vec![]).into());
        }
        let die: Vec<_> = d.sum().into_die_iter().collect();
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
    if let (RuntimeValue::Pool(d), RuntimeValue::List(cond)) = (&args[0], &args[1]) {
        if d.is_empty() {
            return Ok(Pool::from_list(1, vec![]).into());
        }
        let die: Vec<_> = d.sum().into_die_iter().collect();
        Ok(Pool::from(explode(die, cond, explode_depth)).into())
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
    if let RuntimeValue::Pool(d) = &args[0] {
        if d.is_empty() {
            return Ok(Pool::from_list(1, vec![]).into());
        }
        let die: Vec<_> = d.sum().into_die_iter().collect();
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
    if let (RuntimeValue::Pool(d), RuntimeValue::List(cond)) = (&args[0], &args[1]) {
        if d.is_empty() {
            return Ok(Pool::from_list(1, vec![]).into());
        }
        let die: Vec<_> = d.sum().into_die_iter().collect();
        Ok(Pool::from(reroll(die, cond, explode_depth)).into())
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
    if let (RuntimeValue::Int(i), RuntimeValue::Pool(d)) = (&args[0], &args[1]) {
        let keep_list = keep_list_for_primitive(
            PrimitiveType::Highest,
            *i,
            arg_ranges[0],
            d.dimension() as usize,
            function_range,
        )?;
        Ok(d.sum_with_keep_list(&keep_list).into())
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
    if let (RuntimeValue::Int(i), RuntimeValue::Pool(d)) = (&args[0], &args[1]) {
        let keep_list = keep_list_for_primitive(
            PrimitiveType::Lowest,
            *i,
            arg_ranges[0],
            d.dimension() as usize,
            function_range,
        )?;
        Ok(d.sum_with_keep_list(&keep_list).into())
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
    if let (RuntimeValue::Int(i), RuntimeValue::Pool(d)) = (&args[0], &args[1]) {
        let keep_list = keep_list_for_primitive(
            PrimitiveType::Middle,
            *i,
            arg_ranges[0],
            d.dimension() as usize,
            function_range,
        )?;
        Ok(d.sum_with_keep_list(&keep_list).into())
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
    if let (RuntimeValue::Int(i), RuntimeValue::Int(j)) = (&args[0], &args[1]) {
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
    if let (RuntimeValue::Int(i), RuntimeValue::Int(j)) = (&args[0], &args[1]) {
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
    if let RuntimeValue::Pool(d) = &args[0] {
        Ok(d.sum()
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
    if let (RuntimeValue::Pool(first), RuntimeValue::Int(condition), RuntimeValue::Pool(second)) =
        (&args[0], &args[1], &args[2])
    {
        Ok(if *condition == 0 {
            RuntimeValue::Pool(second.clone())
        } else {
            RuntimeValue::Pool(first.clone())
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
    if let RuntimeValue::List(lst) = &args[0] {
        Ok(lst.iter().rev().copied().collect::<Vec<_>>().into())
    } else {
        panic!("wrong argument types to [reverse]");
    }
}

fn sort_execute(
    args: &[RuntimeValue],
    _arg_ranges: &[ast::Range],
    _explode_depth: usize,
    lowest_first: bool,
    _function_range: ast::Range,
) -> Result<RuntimeValue, crate::eval::RuntimeError> {
    if let RuntimeValue::List(lst) = &args[0] {
        let mut lst = (**lst).clone();
        if lowest_first {
            lst.sort_unstable();
        } else {
            lst.sort_unstable_by_key(|o| -o);
        }
        Ok(lst.into())
    } else {
        panic!("wrong argument types to [sort]");
    }
}

lazy_static! {
    pub static ref ABSOLUTE_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int)],
        execute: absolute_execute,
    };
    pub static ref CONTAINS_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::List), Some(StaticType::Int)],
        execute: contains_execute,
    };
    pub static ref COUNT_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::List), Some(StaticType::List)],
        execute: count_execute,
    };
    pub static ref EXPLODE_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Pool)],
        execute: explode_execute,
    };
    pub static ref HIGHEST_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int), Some(StaticType::Pool)],
        execute: highest_execute,
    };
    pub static ref LOWEST_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int), Some(StaticType::Pool)],
        execute: lowest_execute,
    };
    pub static ref MIDDLE_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int), Some(StaticType::Pool)],
        execute: middle_execute,
    };
    pub static ref HIGHEST_OF_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int), Some(StaticType::Int)],
        execute: highest_of_execute,
    };
    pub static ref LOWEST_OF_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Int), Some(StaticType::Int)],
        execute: lowest_of_execute,
    };
    pub static ref MAXIMUM_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Pool)],
        execute: maximum_execute,
    };
    pub static ref CHOOSE_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![
            Some(StaticType::Pool),
            Some(StaticType::Int),
            Some(StaticType::Pool),
        ],
        execute: choose_execute,
    };
    pub static ref REVERSE_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::List)],
        execute: reverse_execute,
    };
    pub static ref SORT_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::List)],
        execute: sort_execute,
    };
    pub static ref EXPLODE_ON_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Pool), Some(StaticType::List)],
        execute: explode_on_execute,
    };
    pub static ref REROLL_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Pool)],
        execute: reroll_execute,
    };
    pub static ref REROLL_ON_PRIMITIVE: Primitive = Primitive {
        arg_types: vec![Some(StaticType::Pool), Some(StaticType::List)],
        execute: reroll_on_execute,
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
}

#[cfg(test)]
mod tests {
    use malachite::Natural;

    use super::*;
    use crate::{ast::Range, dice::Pool};
    use std::rc::Rc;

    fn dummy_range() -> Range {
        Range { start: 0, end: 0 }
    }

    fn to_nat_list(outcomes: &[(i32, Natural)]) -> Vec<(i32, i32)> {
        outcomes
            .iter()
            .map(|(i, w)| (*i, w.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn test_absolute_execute() {
        let args = vec![RuntimeValue::Int(-5)];
        let result = absolute_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, RuntimeValue::Int(5));

        let args = vec![RuntimeValue::Int(3)];
        let result = absolute_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, RuntimeValue::Int(3));
    }

    #[test]
    fn test_contains_execute() {
        let haystack = vec![1, 2, 3, 3, 4, 5];
        let args = vec![RuntimeValue::List(Rc::new(haystack)), RuntimeValue::Int(3)];
        let result = contains_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, RuntimeValue::Int(1));

        let haystack = vec![1, 2, 4, 5];
        let args = vec![RuntimeValue::List(Rc::new(haystack)), RuntimeValue::Int(3)];
        let result = contains_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, RuntimeValue::Int(0));
    }

    #[test]
    fn test_count_execute() {
        let needle = vec![1, 2, 2];
        let haystack = vec![1, 2, 2, 3, 2, 1];
        let args = vec![
            RuntimeValue::List(Rc::new(needle)),
            RuntimeValue::List(Rc::new(haystack)),
        ];
        let result = count_execute(&args, &[], 0, false, dummy_range()).unwrap();
        // Should count: 1 appears 2 times, 2 appears 3 times -> 1*2 + 2*3 = 8
        assert_eq!(result, RuntimeValue::Int(8));
    }

    #[test]
    fn test_explode_execute() {
        let pool = Pool::from_list(1, vec![1, 2, 3]);
        let args = vec![RuntimeValue::Pool(Rc::new(pool))];
        let result = explode_execute(&args, &[], 2, false, dummy_range()).unwrap();
        // Should return a pool where 3s explode
        if let RuntimeValue::Pool(result_pool) = result {
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
        let args = vec![RuntimeValue::Int(2), RuntimeValue::Pool(Rc::new(pool))];
        let ranges = vec![dummy_range(), dummy_range()];
        let result = highest_execute(&args, &ranges, 0, false, dummy_range()).unwrap();
        // Should keep the highest 2 dice from a 3d6 roll
        if let RuntimeValue::Pool(result_pool) = result {
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
        let args = vec![RuntimeValue::Int(2), RuntimeValue::Pool(Rc::new(pool))];
        let ranges = vec![dummy_range(), dummy_range()];
        let result = lowest_execute(&args, &ranges, 0, false, dummy_range()).unwrap();
        // Should keep the lowest 2 dice from a 3d6 roll
        if let RuntimeValue::Pool(result_pool) = result {
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
        let args = vec![RuntimeValue::Int(1), RuntimeValue::Pool(Rc::new(pool))];
        let ranges = vec![dummy_range(), dummy_range()];
        let result = middle_execute(&args, &ranges, 0, false, dummy_range()).unwrap();
        // Should keep the middle 1 die from a 3d6 roll
        if let RuntimeValue::Pool(result_pool) = result {
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
        let args = vec![RuntimeValue::Int(5), RuntimeValue::Int(3)];
        let result = highest_of_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, RuntimeValue::Int(5));

        let args = vec![RuntimeValue::Int(2), RuntimeValue::Int(7)];
        let result = highest_of_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, RuntimeValue::Int(7));
    }

    #[test]
    fn test_lowest_of_execute() {
        let args = vec![RuntimeValue::Int(5), RuntimeValue::Int(3)];
        let result = lowest_of_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, RuntimeValue::Int(3));

        let args = vec![RuntimeValue::Int(2), RuntimeValue::Int(7)];
        let result = lowest_of_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, RuntimeValue::Int(2));
    }

    #[test]
    fn test_maximum_execute() {
        let pool = Pool::from_list(1, vec![1, 3, 2]);
        let args = vec![RuntimeValue::Pool(Rc::new(pool))];
        let result = maximum_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, RuntimeValue::Int(3));

        // Test empty pool
        let empty_pool = Pool::from_list(1, vec![]);
        let args = vec![RuntimeValue::Pool(Rc::new(empty_pool))];
        let result = maximum_execute(&args, &[], 0, false, dummy_range()).unwrap();
        assert_eq!(result, RuntimeValue::Int(0));
    }

    #[test]
    fn test_choose_execute() {
        let first = RuntimeValue::Pool(Rc::new(Pool::ndn(2, 6)));
        let second = RuntimeValue::Pool(Rc::new(Pool::ndn(1, 20)));

        let result = choose_execute(
            &[first.clone(), RuntimeValue::Int(-1), second.clone()],
            &[],
            0,
            false,
            dummy_range(),
        )
        .unwrap();
        assert_eq!(result, first);

        let result = choose_execute(
            &[first, RuntimeValue::Int(0), second.clone()],
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
        let [(RuntimeValue::Pool(result), _)] = outputs.as_slice() else {
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
        let args = vec![RuntimeValue::List(Rc::new(list))];
        let result = reverse_execute(&args, &[], 0, false, dummy_range()).unwrap();
        if let RuntimeValue::List(reversed) = result {
            assert_eq!(*reversed, vec![5, 4, 3, 2, 1]);
        } else {
            panic!("Expected list result");
        }
    }

    #[test]
    fn test_sort_execute() {
        let list = vec![3, 1, 4, 1, 5];
        let args = vec![RuntimeValue::List(Rc::new(list))];

        // Test lowest first (ascending)
        let result = sort_execute(&args, &[], 0, true, dummy_range()).unwrap();
        if let RuntimeValue::List(sorted) = result {
            assert_eq!(*sorted, vec![1, 1, 3, 4, 5]);
        } else {
            panic!("Expected list result");
        }

        // Test highest first (descending)
        let list = vec![3, 1, 4, 1, 5];
        let args = vec![RuntimeValue::List(Rc::new(list))];
        let result = sort_execute(&args, &[], 0, false, dummy_range()).unwrap();
        if let RuntimeValue::List(sorted) = result {
            assert_eq!(*sorted, vec![5, 4, 3, 1, 1]);
        } else {
            panic!("Expected list result");
        }
    }

    #[test]
    fn test_explode_on_execute() {
        let pool = Pool::from_list(1, vec![1, 2, 3]);
        let condition = vec![2, 3];
        let args = vec![
            RuntimeValue::Pool(Rc::new(pool)),
            RuntimeValue::List(Rc::new(condition)),
        ];
        let result = explode_on_execute(&args, &[], 2, false, dummy_range()).unwrap();
        // Should return a pool where 2s and 3s explode
        if let RuntimeValue::Pool(result_pool) = result {
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
        let args = vec![RuntimeValue::Pool(Rc::new(pool))];
        let result = reroll_execute(&args, &[], 2, false, dummy_range()).unwrap();
        // Should return a pool where 3s reroll (without keeping the original value)
        if let RuntimeValue::Pool(result_pool) = result {
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
        let args = vec![
            RuntimeValue::Pool(Rc::new(pool)),
            RuntimeValue::List(Rc::new(condition)),
        ];
        let result = reroll_on_execute(&args, &[], 2, false, dummy_range()).unwrap();
        // Should return a pool where 1s and 4s reroll
        if let RuntimeValue::Pool(result_pool) = result {
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

        // Should have registered exactly 16 functions
        assert_eq!(functions.len(), 16);
    }
}
