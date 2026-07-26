//! Unary, binary, positional, and dice-construction operator semantics.

use std::rc::Rc;

use crate::{
    ast::{self, BinaryOp, UnaryOp, WithRange},
    dice::Pool,
    error::RuntimeError,
    value::{ElementType, ElementValue, RuntimeValue, expect_int, sum_elements, sum_pool},
};

pub(crate) fn apply_unary_op(
    op: &WithRange<UnaryOp>,
    operand: &RuntimeValue,
) -> Result<RuntimeValue, RuntimeError> {
    match op.value {
        UnaryOp::D => make_d(None, operand, op.range),
        UnaryOp::Negate if !operand.is_additive() => Err(RuntimeError::EnumTypeError {
            range: op.range.into(),
            message: format!("operator {} is not defined for this element type", op.value),
        }),
        UnaryOp::Invert if !operand.is_numeric_compatible() => Err(RuntimeError::EnumTypeError {
            range: op.range.into(),
            message: format!("operator {} is not defined for this element type", op.value),
        }),
        UnaryOp::Negate => negate_value(operand).map_err(|message| RuntimeError::MathError {
            range: op.range.into(),
            message,
        }),
        UnaryOp::Invert => Ok(operand.map_numeric_outcomes(|o| if o == 0 { 1 } else { 0 })),
        UnaryOp::Length => Ok(match operand {
            RuntimeValue::Element(ElementValue::AdditiveIdentity) => 1.into(),
            RuntimeValue::Element(ElementValue::Int(i)) => i32::try_from(i.abs().to_string().len())
                .expect("vector length fits in i32")
                .into(),
            RuntimeValue::Element(ElementValue::Enum { .. }) => 1.into(),
            RuntimeValue::Element(ElementValue::Tuple(fields)) => i32::try_from(fields.len())
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

fn negate_value(value: &RuntimeValue) -> Result<RuntimeValue, String> {
    match value {
        RuntimeValue::Element(value) => Ok(RuntimeValue::Element(value.checked_neg()?)),
        RuntimeValue::List(values, outcome_type) => Ok(RuntimeValue::Element(
            sum_elements(values, outcome_type).checked_neg()?,
        )),
        RuntimeValue::Pool(pool, outcome_type) => Ok(RuntimeValue::Pool(
            Rc::new(
                (**pool)
                    .clone()
                    .try_map_outcomes(|outcome| outcome.checked_neg())?,
            ),
            outcome_type.clone(),
        )),
    }
}

fn math_result_type(op: BinaryOp, left: &ElementType, right: &ElementType) -> Option<ElementType> {
    let left = left.summed_type();
    let right = right.summed_type();
    match op {
        BinaryOp::Add | BinaryOp::Sub => left.merged_with(&right).filter(ElementType::is_additive),
        BinaryOp::Mul
            if matches!(left, ElementType::Tuple(_))
                && left.is_additive()
                && matches!(right, ElementType::Int | ElementType::AdditiveIdentity) =>
        {
            Some(left)
        }
        BinaryOp::Mul
            if matches!(right, ElementType::Tuple(_))
                && right.is_additive()
                && matches!(left, ElementType::Int | ElementType::AdditiveIdentity) =>
        {
            Some(right)
        }
        BinaryOp::Mul
            if matches!(left, ElementType::Int | ElementType::AdditiveIdentity)
                && matches!(right, ElementType::Int | ElementType::AdditiveIdentity) =>
        {
            left.merged_with(&right)
        }
        BinaryOp::Div
            if matches!(right, ElementType::Int | ElementType::AdditiveIdentity)
                && left.is_additive() =>
        {
            Some(left)
        }
        BinaryOp::Pow | BinaryOp::Or | BinaryOp::And
            if matches!(left, ElementType::Int | ElementType::AdditiveIdentity)
                && matches!(right, ElementType::Int | ElementType::AdditiveIdentity) =>
        {
            Some(ElementType::Int)
        }
        _ => None,
    }
}

fn apply_element_math(
    op: BinaryOp,
    left: &ElementValue,
    right: &ElementValue,
) -> Result<ElementValue, String> {
    match op {
        BinaryOp::Pow => {
            let left = left.materialize_identity(&ElementType::Int);
            let right = right.materialize_identity(&ElementType::Int);
            let (ElementValue::Int(left), ElementValue::Int(right)) = (left, right) else {
                unreachable!("power operands were type checked")
            };
            if right < 0 {
                Err(format!("Cannot raise {} to negative power {}", left, right))
            } else {
                left.checked_pow(right.unsigned_abs())
                    .map(ElementValue::Int)
                    .ok_or_else(|| format!("Power overflow: {} ^ {}", left, right))
            }
        }
        BinaryOp::Add => match (left, right) {
            (ElementValue::AdditiveIdentity, other) | (other, ElementValue::AdditiveIdentity) => {
                Ok(other.clone())
            }
            _ => left.try_zip_ints(right, &|left, right| {
                left.checked_add(right)
                    .ok_or_else(|| format!("Addition overflow: {} + {}", left, right))
            }),
        },
        BinaryOp::Sub => match (left, right) {
            (ElementValue::AdditiveIdentity, ElementValue::AdditiveIdentity) => {
                Ok(ElementValue::AdditiveIdentity)
            }
            (other, ElementValue::AdditiveIdentity) => Ok(other.clone()),
            (ElementValue::AdditiveIdentity, other) => other.checked_neg(),
            _ => left.try_zip_ints(right, &|left, right| {
                left.checked_sub(right)
                    .ok_or_else(|| format!("Subtraction overflow: {} - {}", left, right))
            }),
        },
        BinaryOp::Mul => match (left, right) {
            (ElementValue::AdditiveIdentity, ElementValue::AdditiveIdentity)
            | (ElementValue::AdditiveIdentity, ElementValue::Int(_))
            | (ElementValue::Int(_), ElementValue::AdditiveIdentity) => {
                Ok(ElementValue::AdditiveIdentity)
            }
            (ElementValue::AdditiveIdentity, ElementValue::Tuple(_)) => Ok(right
                .element_type()
                .additive_identity()
                .expect("additive tuple")),
            (ElementValue::Tuple(_), ElementValue::AdditiveIdentity) => Ok(left
                .element_type()
                .additive_identity()
                .expect("additive tuple")),
            (ElementValue::Int(left), ElementValue::Int(right)) => left
                .checked_mul(*right)
                .map(ElementValue::Int)
                .ok_or_else(|| format!("Multiplication overflow: {} * {}", left, right)),
            (ElementValue::Tuple(_), ElementValue::Int(element)) => left.try_map_ints(&|value| {
                value
                    .checked_mul(*element)
                    .ok_or_else(|| format!("Multiplication overflow: {} * {}", value, element))
            }),
            (ElementValue::Int(element), ElementValue::Tuple(_)) => right.try_map_ints(&|value| {
                element
                    .checked_mul(value)
                    .ok_or_else(|| format!("Multiplication overflow: {} * {}", element, value))
            }),
            _ => unreachable!("multiplication operands were type checked"),
        },
        BinaryOp::Div => {
            let divisor = right.materialize_identity(&ElementType::Int);
            let ElementValue::Int(divisor) = divisor else {
                unreachable!("division divisor was type checked")
            };
            if divisor == 0 {
                return Err(format!("Cannot divide {} by zero", left));
            }
            if matches!(left, ElementValue::AdditiveIdentity) {
                return Ok(ElementValue::AdditiveIdentity);
            }
            left.try_map_ints(&|value| {
                value
                    .checked_div(divisor)
                    .ok_or_else(|| format!("Division overflow: {} / {}", value, divisor))
            })
        }
        BinaryOp::Or | BinaryOp::And => {
            let left = left.materialize_identity(&ElementType::Int);
            let right = right.materialize_identity(&ElementType::Int);
            let (ElementValue::Int(left), ElementValue::Int(right)) = (left, right) else {
                unreachable!("logical operands were type checked")
            };
            let result = match op {
                BinaryOp::Or => left != 0 || right != 0,
                BinaryOp::And => left != 0 && right != 0,
                _ => unreachable!(),
            };
            Ok(ElementValue::Int(i32::from(result)))
        }
        _ => unreachable!("non-mathematical operator passed to element math"),
    }
}

fn apply_math_op(
    op: &WithRange<BinaryOp>,
    left: &RuntimeValue,
    right: &RuntimeValue,
) -> Result<RuntimeValue, RuntimeError> {
    let result_type = math_result_type(op.value, &left.outcome_type(), &right.outcome_type())
        .ok_or_else(|| RuntimeError::EnumTypeError {
            range: op.range.into(),
            message: format!(
                "operator {} is not defined for these element types",
                op.value
            ),
        })?;
    let sum_sequence = |operand: &RuntimeValue| match operand {
        RuntimeValue::List(values, outcome_type) => {
            RuntimeValue::Element(sum_elements(values, outcome_type))
        }
        _ => operand.clone(),
    };
    let left = sum_sequence(left);
    let right = sum_sequence(right);
    broadcast_binary(
        &left,
        &right,
        |left, right| apply_element_math(op.value, left, right),
        |_, _| unreachable!("mathematical sequences are summed before broadcasting"),
        result_type,
    )
    .map_err(|message| RuntimeError::MathError {
        range: op.range.into(),
        message,
    })
}

pub(crate) fn apply_binary_op(
    op: &WithRange<BinaryOp>,
    left: &RuntimeValue,
    left_range: ast::Range,
    right: &RuntimeValue,
    lowest_first: bool,
) -> Result<RuntimeValue, RuntimeError> {
    if matches!(
        op.value,
        BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
    ) && (!left.is_numeric_compatible() || !right.is_numeric_compatible())
    {
        return Err(RuntimeError::EnumTypeError {
            range: op.range.into(),
            message: format!(
                "operator {} is not defined for these element types",
                op.value
            ),
        });
    }
    if matches!(op.value, BinaryOp::Eq | BinaryOp::Ne) && left.merged_outcome_type(right).is_none()
    {
        return Err(RuntimeError::EnumTypeError {
            range: op.range.into(),
            message: "equality requires operands with the same outcome type".to_string(),
        });
    }
    match &op.value {
        BinaryOp::D => make_d(Some(left), right, op.range),
        BinaryOp::At => {
            if !left.is_numeric_compatible() {
                return Err(RuntimeError::EnumTypeError {
                    range: op.range.into(),
                    message: "only integers can be used as positions".to_string(),
                });
            }
            let left = match left {
                RuntimeValue::Element(ElementValue::AdditiveIdentity) => Rc::new(vec![0]),
                RuntimeValue::Element(ElementValue::Int(i)) => Rc::new(vec![*i]),
                RuntimeValue::List(lst, _) => {
                    Rc::new(lst.iter().map(expect_int).collect::<Vec<_>>())
                }
                RuntimeValue::Pool(_, _) => {
                    return Err(RuntimeError::InvalidArgumentToOperator {
                        operator_range: op.range.into(),
                        op: op.value,
                        expected: "an integer or a sequence",
                        found_range: left_range.into(),
                        found: left.runtime_type(),
                        value: left.clone(),
                    });
                }
                RuntimeValue::Element(ElementValue::Enum { .. } | ElementValue::Tuple(_)) => {
                    unreachable!()
                }
            };
            match right {
                RuntimeValue::Element(ElementValue::AdditiveIdentity) => {
                    Ok(RuntimeValue::Element(select_positions(
                        &left,
                        &[ElementValue::Int(0)],
                        &ElementType::Int,
                        lowest_first,
                    )))
                }
                RuntimeValue::Element(ElementValue::Int(i)) => {
                    let digits = i
                        .abs()
                        .to_string()
                        .chars()
                        // Unwrap here is ok as all chars for a *positive* integer are valid digits.
                        .map(|c| {
                            ElementValue::Int(
                                i32::try_from(c.to_digit(10).unwrap()).expect("digit fits in i32")
                                    * i.signum(),
                            )
                        })
                        .collect::<Vec<_>>();
                    Ok(RuntimeValue::Element(select_positions(
                        &left,
                        &digits,
                        &ElementType::Int,
                        lowest_first,
                    )))
                }
                RuntimeValue::List(lst, outcome_type) if right.is_additive() => Ok(
                    RuntimeValue::Element(select_positions(&left, lst, outcome_type, false)),
                ),
                RuntimeValue::Pool(p, outcome_type) if right.is_additive() => {
                    Ok(RuntimeValue::Pool(
                        Rc::new(select_in_dice(&left, p, outcome_type, lowest_first)),
                        outcome_type.clone(),
                    ))
                }
                RuntimeValue::List(lst, _) => {
                    if left.len() != 1 {
                        return Err(RuntimeError::EnumTypeError { range: op.range.into(), message: "selecting multiple non-additive positions would require summing their values".to_string() });
                    }
                    let index = left[0];
                    if index < 1 || index > i32::try_from(lst.len()).unwrap_or(i32::MAX) {
                        return Err(RuntimeError::EnumTypeError {
                            range: op.range.into(),
                            message: "position is out of range".to_string(),
                        });
                    }
                    Ok(RuntimeValue::Element(
                        lst[usize::try_from(index - 1).unwrap()].clone(),
                    ))
                }
                RuntimeValue::Element(ElementValue::Enum { .. } | ElementValue::Tuple(_))
                | RuntimeValue::Pool(_, _) => Err(RuntimeError::EnumTypeError {
                    range: op.range.into(),
                    message: "positional selection is not defined for this value".to_string(),
                }),
            }
        }
        BinaryOp::Pow
        | BinaryOp::Add
        | BinaryOp::Sub
        | BinaryOp::Mul
        | BinaryOp::Div
        | BinaryOp::Or
        | BinaryOp::And => apply_math_op(op, left, right),
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
    }
}

fn select_positions(
    indices: &[i32],
    values: &[ElementValue],
    outcome_type: &ElementType,
    lowest_first: bool,
) -> ElementValue {
    indices.iter().fold(
        outcome_type
            .additive_identity()
            .expect("position selection requires additive outcomes"),
        |sum, &index| {
            let selected = index
                .checked_sub(1)
                .and_then(|index| usize::try_from(index).ok())
                .filter(|&index| index < values.len())
                .map(|index| {
                    if lowest_first {
                        values.len() - index - 1
                    } else {
                        index
                    }
                })
                .and_then(|index| values.get(index));
            selected.map_or(sum.clone(), |value| sum.add_scaled(value, 1))
        },
    )
}

fn select_in_dice(
    indices: &[i32],
    pool: &Pool<ElementValue>,
    outcome_type: &ElementType,
    lowest_first: bool,
) -> Pool<ElementValue> {
    let dimension = usize::try_from(pool.dimension()).expect("usize is at least 32 bits");
    let mut keep_list = vec![false; dimension];
    for &index in indices {
        if index < 1 || index > i32::try_from(dimension).expect("pool dimension fits in i32") {
            continue;
        }
        let index = usize::try_from(index).expect("positive i32 fits in usize");
        let index = if !lowest_first {
            dimension - index
        } else {
            index - 1
        };
        keep_list[index] = true;
    }
    pool.sum_with_keep_list_by(
        &keep_list,
        outcome_type
            .additive_identity()
            .expect("position selection requires additive outcomes"),
        ElementValue::add_scaled,
    )
}

/// Broadcasts an element operation over runtime operand shapes.
///
/// Sequence/sequence behavior is operator-specific. Sequence/element pairs
/// apply the element operation to every sequence member and sum the results.
/// If either operand is a pool, both operands are first converted to summed
/// distributions and their cross product is combined into a new distribution.
fn broadcast_binary<E>(
    left: &RuntimeValue,
    right: &RuntimeValue,
    element_op: impl Fn(&ElementValue, &ElementValue) -> Result<ElementValue, E>,
    list_list_op: impl Fn(&[ElementValue], &[ElementValue]) -> Result<ElementValue, E>,
    result_type: ElementType,
) -> Result<RuntimeValue, E> {
    match (left, right) {
        (RuntimeValue::Element(left), RuntimeValue::Element(right)) => {
            return Ok(RuntimeValue::Element(element_op(left, right)?));
        }
        (RuntimeValue::List(left, _), RuntimeValue::List(right, _)) => {
            return Ok(RuntimeValue::Element(list_list_op(left, right)?));
        }
        (RuntimeValue::List(left, _), RuntimeValue::Element(right)) => {
            let results = left
                .iter()
                .map(|left| element_op(left, right))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(RuntimeValue::Element(sum_elements(&results, &result_type)));
        }
        (RuntimeValue::Element(left), RuntimeValue::List(right, _)) => {
            let results = right
                .iter()
                .map(|right| element_op(left, right))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(RuntimeValue::Element(sum_elements(&results, &result_type)));
        }
        (RuntimeValue::Pool(_, _), _) | (_, RuntimeValue::Pool(_, _)) => {}
    }

    let left_pool = sum_pool(&left.to_pool(), &left.outcome_type());
    let right_pool = sum_pool(&right.to_pool(), &right.outcome_type());
    let components = left_pool
        .ordered_outcomes()
        .iter()
        .map(|(left_outcome, left_weight)| {
            let distribution = right_pool
                .clone()
                .try_map_outcomes(|right_outcome| element_op(left_outcome, &right_outcome))?;
            Ok((left_weight.clone(), distribution))
        })
        .collect::<Result<Vec<_>, E>>()?;

    Ok(RuntimeValue::Pool(
        Rc::new(Pool::from_mixture(components)),
        result_type,
    ))
}

fn comp_binary_op(
    left: &RuntimeValue,
    right: &RuntimeValue,
    int_comp: impl Fn(i32, i32) -> i32,
    list_comp: impl Fn(&[i32], &[i32]) -> i32,
) -> RuntimeValue {
    let left = left.materialize_identities(&ElementType::Int);
    let right = right.materialize_identities(&ElementType::Int);
    broadcast_binary(
        &left,
        &right,
        |left, right| {
            Ok::<_, std::convert::Infallible>(ElementValue::Int(int_comp(
                expect_int(left),
                expect_int(right),
            )))
        },
        |left, right| {
            Ok(ElementValue::Int(list_comp(
                &left.iter().map(expect_int).collect::<Vec<_>>(),
                &right.iter().map(expect_int).collect::<Vec<_>>(),
            )))
        },
        ElementType::Int,
    )
    .expect("comparison broadcasting is infallible")
}

fn equality_binary_op(left: &RuntimeValue, right: &RuntimeValue, equal: bool) -> RuntimeValue {
    let outcome_type = left
        .merged_outcome_type(right)
        .expect("equality operand types were checked")
        .summed_type();
    let left = left.materialize_identities(&outcome_type);
    let right = right.materialize_identities(&outcome_type);
    let compare = |a: &ElementValue, b: &ElementValue| i32::from((a == b) == equal);
    broadcast_binary(
        &left,
        &right,
        |left, right| Ok::<_, std::convert::Infallible>(ElementValue::Int(compare(left, right))),
        |left, right| Ok(ElementValue::Int(i32::from((left == right) == equal))),
        ElementType::Int,
    )
    .expect("equality broadcasting is infallible")
}

enum DiceCount {
    Int(i32),
    Pool(Rc<Pool<i32>>),
}

enum DRightSide {
    List(Vec<ElementValue>),
    Pool(Rc<Pool<ElementValue>>),
}

fn normalize_dice_count(arg: &RuntimeValue) -> DiceCount {
    match arg {
        RuntimeValue::Element(ElementValue::AdditiveIdentity) => DiceCount::Int(0),
        RuntimeValue::Element(ElementValue::Int(i)) => DiceCount::Int(*i),
        RuntimeValue::List(list, _) => DiceCount::Int(list.iter().map(expect_int).sum()),
        RuntimeValue::Pool(pool, _) => DiceCount::Pool(Rc::new(
            (**pool)
                .clone()
                .map_outcomes(|outcome| expect_int(&outcome)),
        )),
        RuntimeValue::Element(ElementValue::Enum { .. } | ElementValue::Tuple(_)) => {
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
    if left.is_some_and(|value| !value.is_numeric_compatible()) {
        return Err(RuntimeError::EnumTypeError {
            range: range.into(),
            message: "only numeric values can be used as dice counts".to_string(),
        });
    }
    let repeat = left.map_or(DiceCount::Int(1), normalize_dice_count);
    let outcome_type = right.outcome_type();
    let right = match right {
        RuntimeValue::Element(ElementValue::AdditiveIdentity) => {
            DRightSide::List(vec![ElementValue::Int(0)])
        }
        RuntimeValue::Element(ElementValue::Int(sides)) => {
            if *sides > 0 {
                DRightSide::List((1..=*sides).map(ElementValue::Int).collect())
            } else if *sides == 0 {
                DRightSide::List(vec![ElementValue::Int(0)])
            } else {
                DRightSide::List((*sides..=-1).map(ElementValue::Int).collect())
            }
        }
        RuntimeValue::List(list, _) => DRightSide::List((**list).clone()),
        RuntimeValue::Pool(d, _) => DRightSide::Pool(Rc::clone(d)),
        RuntimeValue::Element(ElementValue::Enum { .. } | ElementValue::Tuple(_)) => {
            return Err(RuntimeError::EnumTypeError {
                range: range.into(),
                message: "a non-numeric element cannot specify die sides; use a sequence"
                    .to_string(),
            });
        }
    };
    if !outcome_type.is_additive() && !matches!(&repeat, DiceCount::Int(1)) {
        return Err(RuntimeError::EnumTypeError {
            range: range.into(),
            message: "non-additive pools must have dimension one".to_string(),
        });
    }
    let result: RuntimeValue = match (repeat, right) {
        (DiceCount::Int(i), DRightSide::List(list)) => RuntimeValue::Pool(
            Rc::new(
                make_pool(i, list).map_err(|message| RuntimeError::MathError {
                    range: range.into(),
                    message,
                })?,
            ),
            outcome_type.clone(),
        ),
        (DiceCount::Int(i), DRightSide::Pool(p)) => {
            let mut new_pool = (*p).clone();
            if i < 0 {
                new_pool = new_pool
                    .try_map_outcomes(|outcome| outcome.checked_neg())
                    .map_err(|message| RuntimeError::MathError {
                        range: range.into(),
                        message,
                    })?;
            }
            new_pool.set_dimension(new_pool.dimension() * i.unsigned_abs());
            RuntimeValue::Pool(Rc::new(new_pool), outcome_type.clone())
        }
        (DiceCount::Pool(left_p), right) => {
            let left_p = (*left_p).sum();
            let right = match right {
                DRightSide::List(list) => Pool::from_list(1, list),
                DRightSide::Pool(p) => sum_pool(&p, &outcome_type),
            };
            let negated_right = if left_p
                .ordered_outcomes()
                .iter()
                .any(|(multiplier, _)| *multiplier < 0)
            {
                Some(
                    right
                        .clone()
                        .try_map_outcomes(|outcome| outcome.checked_neg())
                        .map_err(|message| RuntimeError::MathError {
                            range: range.into(),
                            message,
                        })?,
                )
            } else {
                None
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
                    let multiplier = count[0];
                    let mut dup_right = if multiplier < 0 {
                        negated_right
                            .clone()
                            .expect("negative outcomes have a negated right-hand pool")
                    } else {
                        right.clone()
                    };
                    dup_right.set_dimension(dup_right.dimension() * multiplier.unsigned_abs());
                    sum_pool(&dup_right, &outcome_type).into()
                })),
                outcome_type.clone(),
            )
        }
    };
    Ok(result)
}

fn make_pool(n: i32, sides: Vec<ElementValue>) -> Result<Pool<ElementValue>, String> {
    let sides = if n < 0 {
        sides
            .into_iter()
            .map(|side| side.checked_neg())
            .collect::<Result<Vec<_>, _>>()?
    } else {
        sides
    };
    let dimension = if sides.is_empty() {
        0
    } else {
        n.unsigned_abs()
    };
    Ok(Pool::from_list(dimension, sides))
}
