//! Unary, binary, positional, and dice-construction operator semantics.

use std::rc::Rc;

use crate::{
    ast::{self, BinaryOp, UnaryOp, WithRange},
    dice::Pool,
    error::{NonAdditiveSubject, RuntimeError, SemanticErrorKind},
    value::{
        ElementMismatch, ElementOpError, ElementValue, PoolSumFailure, RuntimeValue, sum_elements,
        sum_pool,
    },
};

/// Turns a failed element operation into a runtime error.
///
/// Arithmetic that overflowed is the operation's own fault and reads as a math
/// error; values the operation is not defined for name the value at fault.
fn op_error_at(range: ast::Range, action: &'static str) -> impl Fn(ElementOpError) -> RuntimeError {
    move |error| match error {
        ElementOpError::Math(message) => RuntimeError::MathError { range, message },
        ElementOpError::Mismatch(mismatch) => {
            mismatch.into_error(range, action, NonAdditiveSubject::Operand)
        }
    }
}

fn op_error<T>(op: &WithRange<T>, action: &'static str) -> impl Fn(ElementOpError) -> RuntimeError {
    op_error_at(op.range, action)
}

pub(crate) fn apply_unary_op(
    op: &WithRange<UnaryOp>,
    operand: &RuntimeValue,
) -> Result<RuntimeValue, RuntimeError> {
    match op.value {
        UnaryOp::D => make_d(None, operand, op.range),
        UnaryOp::Negate => negate_value(operand).map_err(op_error(op, "negating a value")),
        UnaryOp::Invert => operand
            .map_numeric_outcomes(|o| if o == 0 { 1 } else { 0 })
            .map_err(|mismatch| {
                mismatch.into_error(op.range, "inverting a value", NonAdditiveSubject::Operand)
            }),
        UnaryOp::Length => Ok(match operand {
            RuntimeValue::Element(ElementValue::AdditiveIdentity) => 1.into(),
            RuntimeValue::Element(ElementValue::Int(i)) => i32::try_from(i.abs().to_string().len())
                .expect("vector length fits in i32")
                .into(),
            RuntimeValue::Element(ElementValue::Symbol(_)) => 1.into(),
            RuntimeValue::Element(ElementValue::Tuple(fields)) => i32::try_from(fields.len())
                .expect("tuple length fits in i32")
                .into(),
            RuntimeValue::List(list) => i32::try_from(list.len())
                .expect("vector length fits in i32")
                .into(),
            RuntimeValue::Pool(d) => i32::try_from(d.dimension())
                .expect("vector length fits in i32")
                .into(),
        }),
    }
}

fn negate_value(value: &RuntimeValue) -> Result<RuntimeValue, ElementOpError> {
    match value {
        RuntimeValue::Element(value) => Ok(RuntimeValue::Element(value.checked_neg()?)),
        RuntimeValue::List(values) => {
            Ok(RuntimeValue::Element(sum_elements(values)?.checked_neg()?))
        }
        RuntimeValue::Pool(pool) => Ok(RuntimeValue::Pool(Rc::new(
            (**pool)
                .clone()
                .try_map_outcomes(|outcome| outcome.checked_neg())?,
        ))),
    }
}

/// The integer an operand must be for the operators that only accept numbers.
fn require_int(value: &ElementValue) -> Result<i32, ElementOpError> {
    value
        .as_int_or_identity()
        .ok_or_else(|| ElementMismatch::not_numeric(value).into())
}

fn apply_element_math(
    op: BinaryOp,
    left: &ElementValue,
    right: &ElementValue,
) -> Result<ElementValue, ElementOpError> {
    match op {
        BinaryOp::Pow => {
            let (left, right) = (require_int(left)?, require_int(right)?);
            if right < 0 {
                Err(format!("Cannot raise {} to negative power {}", left, right).into())
            } else {
                left.checked_pow(right.unsigned_abs())
                    .map(ElementValue::Int)
                    .ok_or_else(|| format!("Power overflow: {} ^ {}", left, right).into())
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
            // Scaling a tuple by the empty sum zeroes it, which needs the
            // tuple's own shape and so fails if any field is not a number.
            (ElementValue::AdditiveIdentity, tuple @ ElementValue::Tuple(_))
            | (tuple @ ElementValue::Tuple(_), ElementValue::AdditiveIdentity) => tuple
                .additive_identity()
                .ok_or_else(|| ElementMismatch::not_numeric(tuple).into()),
            (ElementValue::Int(left), ElementValue::Int(right)) => left
                .checked_mul(*right)
                .map(ElementValue::Int)
                .ok_or_else(|| format!("Multiplication overflow: {} * {}", left, right).into()),
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
            // Two tuples have no product, and a symbol has none either.
            (left, _) if !left.is_additive() => Err(ElementMismatch::not_numeric(left).into()),
            (_, right) if !right.is_additive() => Err(ElementMismatch::not_numeric(right).into()),
            (left, right) => Err(ElementMismatch::shapes(left, right).into()),
        },
        BinaryOp::Div => {
            let divisor = require_int(right)?;
            if divisor == 0 {
                return Err(match left {
                    ElementValue::Int(value) => format!("Cannot divide {value} by zero"),
                    _ => "Cannot divide by zero".to_string(),
                }
                .into());
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
            let (left, right) = (require_int(left)?, require_int(right)?);
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
    // A sequence is one value where it meets arithmetic, so it is summed
    // first; a sequence that cannot be summed says so here, at its own values.
    let sum_sequence = |operand: &RuntimeValue| match operand {
        RuntimeValue::List(values) => {
            sum_elements(values)
                .map(RuntimeValue::Element)
                .map_err(|mismatch| {
                    mismatch.into_error(
                        op.range,
                        "this operator",
                        NonAdditiveSubject::Sequence(values.len()),
                    )
                })
        }
        _ => Ok(operand.clone()),
    };
    let left = sum_sequence(left)?;
    let right = sum_sequence(right)?;
    broadcast_binary(
        &left,
        &right,
        |left, right| {
            apply_element_math(op.value, left, right).map_err(op_error(op, "this operator"))
        },
        |_, _| unreachable!("mathematical sequences are summed before broadcasting"),
        op.range,
    )
}

pub(crate) fn apply_binary_op(
    op: &WithRange<BinaryOp>,
    left: &RuntimeValue,
    left_range: ast::Range,
    right: &RuntimeValue,
    lowest_first: bool,
) -> Result<RuntimeValue, RuntimeError> {
    match &op.value {
        BinaryOp::D => make_d(Some(left), right, op.range),
        BinaryOp::At => {
            if left.non_numeric().is_some() {
                return Err(RuntimeError::Semantic {
                    kind: SemanticErrorKind::OperatorOperands,
                    range: op.range,
                    message: "only integers can be used as positions".to_string(),
                });
            }
            let left = match left {
                RuntimeValue::Element(value) => Rc::new(vec![
                    value
                        .as_int_or_identity()
                        .expect("positions were checked to be numbers"),
                ]),
                RuntimeValue::List(lst) => Rc::new(
                    lst.iter()
                        .map(|value| {
                            value
                                .as_int_or_identity()
                                .expect("positions were checked to be numbers")
                        })
                        .collect::<Vec<_>>(),
                ),
                RuntimeValue::Pool(_) => {
                    return Err(RuntimeError::InvalidArgumentToOperator {
                        operator_range: op.range,
                        op: op.value,
                        expected: "an integer or a sequence",
                        found_range: left_range,
                        value: left.clone(),
                    });
                }
            };
            let sum_failed = |mismatch: ElementMismatch, len: usize| {
                mismatch.into_error(
                    op.range,
                    "selecting several positions",
                    NonAdditiveSubject::Sequence(len),
                )
            };
            match right {
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
                    select_positions(&left, &digits, lowest_first)
                        .map(RuntimeValue::Element)
                        .map_err(|mismatch| sum_failed(mismatch, digits.len()))
                }
                RuntimeValue::Element(ElementValue::AdditiveIdentity) => {
                    select_positions(&left, &[ElementValue::Int(0)], lowest_first)
                        .map(RuntimeValue::Element)
                        .map_err(|mismatch| sum_failed(mismatch, 1))
                }
                // One position needs no addition, so it can pick out a value
                // that could never have been summed with anything.
                RuntimeValue::List(lst) if left.len() == 1 => {
                    let index = left[0];
                    if index < 1 || index > i32::try_from(lst.len()).unwrap_or(i32::MAX) {
                        return Err(RuntimeError::Semantic {
                            kind: SemanticErrorKind::OutOfRange,
                            range: op.range,
                            message: "position is out of range".to_string(),
                        });
                    }
                    Ok(RuntimeValue::Element(
                        lst[usize::try_from(index - 1).unwrap()].clone(),
                    ))
                }
                RuntimeValue::List(lst) => select_positions(&left, lst, false)
                    .map(RuntimeValue::Element)
                    .map_err(|mismatch| sum_failed(mismatch, lst.len())),
                RuntimeValue::Pool(p) => select_in_dice(&left, p, lowest_first)
                    .map(|pool| RuntimeValue::Pool(Rc::new(pool)))
                    .map_err(|mismatch| {
                        mismatch.into_error(
                            op.range,
                            "selecting several positions",
                            NonAdditiveSubject::Pool(p.dimension()),
                        )
                    }),
                RuntimeValue::Element(ElementValue::Symbol(_) | ElementValue::Tuple(_)) => {
                    Err(RuntimeError::Semantic {
                        kind: SemanticErrorKind::OperatorOperands,
                        range: op.range,
                        message: "positional selection is not defined for this value".to_string(),
                    })
                }
            }
        }
        BinaryOp::Pow
        | BinaryOp::Add
        | BinaryOp::Sub
        | BinaryOp::Mul
        | BinaryOp::Div
        | BinaryOp::Or
        | BinaryOp::And => apply_math_op(op, left, right),
        BinaryOp::Eq => equality_binary_op(left, right, true, op.range),
        BinaryOp::Ne => equality_binary_op(left, right, false, op.range),
        BinaryOp::Lt => comp_binary_op(
            left,
            right,
            |a, b| if a < b { 1 } else { 0 },
            |a, b| if a < b { 1 } else { 0 },
            op.range,
        ),
        BinaryOp::Le => comp_binary_op(
            left,
            right,
            |a, b| if a <= b { 1 } else { 0 },
            |a, b| if a <= b { 1 } else { 0 },
            op.range,
        ),
        BinaryOp::Gt => comp_binary_op(
            left,
            right,
            |a, b| if a > b { 1 } else { 0 },
            |a, b| if a > b { 1 } else { 0 },
            op.range,
        ),
        BinaryOp::Ge => comp_binary_op(
            left,
            right,
            |a, b| if a >= b { 1 } else { 0 },
            |a, b| if a >= b { 1 } else { 0 },
            op.range,
        ),
    }
}

/// Sums the values at `indices`, seeded with the identity so the first one
/// selected settles the shape. Out-of-range indices select nothing.
fn select_positions(
    indices: &[i32],
    values: &[ElementValue],
    lowest_first: bool,
) -> Result<ElementValue, ElementMismatch> {
    indices
        .iter()
        .try_fold(ElementValue::AdditiveIdentity, |sum, &index| {
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
            match selected {
                Some(value) => sum.add_scaled(value, 1),
                None => Ok(sum),
            }
        })
}

fn select_in_dice(
    indices: &[i32],
    pool: &Pool<ElementValue>,
    lowest_first: bool,
) -> Result<Pool<ElementValue>, ElementMismatch> {
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
    pool.try_sum_with_keep_list_by(
        &keep_list,
        ElementValue::AdditiveIdentity,
        ElementValue::add_scaled,
    )
}

/// Broadcasts an element operation over runtime operand shapes.
///
/// Sequence/sequence behavior is operator-specific. Sequence/element pairs
/// apply the element operation to every sequence member and sum the results.
/// If either operand is a pool, both operands are first converted to summed
/// distributions and their cross product is combined into a new distribution.
fn broadcast_binary(
    left: &RuntimeValue,
    right: &RuntimeValue,
    element_op: impl Fn(&ElementValue, &ElementValue) -> Result<ElementValue, RuntimeError>,
    list_list_op: impl Fn(&[ElementValue], &[ElementValue]) -> Result<ElementValue, RuntimeError>,
    range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    // Broadcasting over a sequence sums the results, so the results have to be
    // summable; that is discovered here rather than predicted beforehand.
    let sum_results = |results: &[ElementValue]| {
        sum_elements(results).map_err(|mismatch| {
            mismatch.into_error(
                range,
                "combining the results of an operator over a sequence",
                NonAdditiveSubject::Sequence(results.len()),
            )
        })
    };
    match (left, right) {
        (RuntimeValue::Element(left), RuntimeValue::Element(right)) => {
            return Ok(RuntimeValue::Element(element_op(left, right)?));
        }
        (RuntimeValue::List(left), RuntimeValue::List(right)) => {
            return Ok(RuntimeValue::Element(list_list_op(left, right)?));
        }
        (RuntimeValue::List(left), RuntimeValue::Element(right)) => {
            let results = left
                .iter()
                .map(|left| element_op(left, right))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(RuntimeValue::Element(sum_results(&results)?));
        }
        (RuntimeValue::Element(left), RuntimeValue::List(right)) => {
            let results = right
                .iter()
                .map(|right| element_op(left, right))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(RuntimeValue::Element(sum_results(&results)?));
        }
        (RuntimeValue::Pool(_), _) | (_, RuntimeValue::Pool(_)) => {}
    }

    let to_summed_pool = |value: &RuntimeValue| {
        let pool = value.to_pool().map_err(|mismatch| {
            mismatch.into_error(
                range,
                "comparing pools",
                NonAdditiveSubject::Sequence(value.elements().len()),
            )
        })?;
        sum_pool(&pool).map_err(|error| error.into_error(range, "comparing pools"))
    };
    let left_pool = to_summed_pool(left)?;
    let right_pool = to_summed_pool(right)?;
    let components = left_pool
        .ordered_outcomes()
        .iter()
        .map(|(left_outcome, left_weight)| {
            let distribution = right_pool
                .clone()
                .try_map_outcomes(|right_outcome| element_op(left_outcome, &right_outcome))?;
            Ok((left_weight.clone(), distribution))
        })
        .collect::<Result<Vec<_>, RuntimeError>>()?;

    Ok(RuntimeValue::Pool(Rc::new(Pool::from_mixture(components))))
}

fn comp_binary_op(
    left: &RuntimeValue,
    right: &RuntimeValue,
    int_comp: impl Fn(i32, i32) -> i32,
    list_comp: impl Fn(&[i32], &[i32]) -> i32,
    range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    // Ordering is only defined on numbers, so each value is asked for one as it
    // is reached.
    let number = |value: &ElementValue| {
        value
            .as_int_or_identity()
            .ok_or_else(|| ElementMismatch::not_numeric(value))
            .map_err(|mismatch| {
                mismatch.into_error(range, "comparing values", NonAdditiveSubject::Operand)
            })
    };
    let numbers =
        |values: &[ElementValue]| values.iter().map(number).collect::<Result<Vec<_>, _>>();
    broadcast_binary(
        left,
        right,
        |left, right| Ok(ElementValue::Int(int_comp(number(left)?, number(right)?))),
        |left, right| {
            Ok(ElementValue::Int(list_comp(
                &numbers(left)?,
                &numbers(right)?,
            )))
        },
        range,
    )
}

fn equality_binary_op(
    left: &RuntimeValue,
    right: &RuntimeValue,
    equal: bool,
    range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    // Equality is total: values of different kinds are simply never equal, so
    // nothing here can fail. The empty sum is the one value that compares
    // across shapes, standing for the zero of whatever it is held against.
    let compare =
        |a: &ElementValue, b: &ElementValue| i32::from(a.equals_with_identity(b) == equal);
    broadcast_binary(
        left,
        right,
        |left, right| Ok(ElementValue::Int(compare(left, right))),
        |left, right| {
            Ok(ElementValue::Int(i32::from(
                (left.len() == right.len()
                    && left
                        .iter()
                        .zip(right.iter())
                        .all(|(left, right)| left.equals_with_identity(right)))
                    == equal,
            )))
        },
        range,
    )
}

enum DiceCount {
    Int(i32),
    Pool(Rc<Pool<i32>>),
}

enum DRightSide {
    List(Vec<ElementValue>),
    Pool(Rc<Pool<ElementValue>>),
}

/// A dice count must be a number; this is where one that is not says so.
fn normalize_dice_count(arg: &RuntimeValue) -> Result<DiceCount, ElementMismatch> {
    let number = |value: &ElementValue| {
        value
            .as_int_or_identity()
            .ok_or_else(|| ElementMismatch::not_numeric(value))
    };
    match arg {
        RuntimeValue::Element(value) => Ok(DiceCount::Int(number(value)?)),
        RuntimeValue::List(list) => Ok(DiceCount::Int(
            list.iter()
                .map(number)
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .sum(),
        )),
        RuntimeValue::Pool(pool) => Ok(DiceCount::Pool(Rc::new(
            (**pool)
                .clone()
                .try_map_outcomes(|outcome| number(&outcome))?,
        ))),
    }
}

/// Executes the unary or binary version of the _n_ `d` _m_ operator.
fn make_d(
    left: Option<&RuntimeValue>,
    right: &RuntimeValue,
    range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    let repeat = match left {
        Some(left) => normalize_dice_count(left).map_err(|mismatch| {
            mismatch.into_error(
                range,
                "using a value as a dice count",
                NonAdditiveSubject::Sequence(left.elements().len()),
            )
        })?,
        None => DiceCount::Int(1),
    };
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
        RuntimeValue::List(list) => DRightSide::List((**list).clone()),
        RuntimeValue::Pool(d) => DRightSide::Pool(Rc::clone(d)),
        RuntimeValue::Element(ElementValue::Symbol(_) | ElementValue::Tuple(_)) => {
            return Err(RuntimeError::Semantic {
                kind: SemanticErrorKind::OperatorOperands,
                range,
                message: "a non-numeric element cannot specify die sides; use a sequence"
                    .to_string(),
            });
        }
    };
    // A negative dice count negates every face; a face that cannot be negated
    // says so from inside `checked_neg`, rather than being predicted here.
    let negate = |mismatch: ElementOpError| op_error_at(range, "negating a die's faces")(mismatch);
    let result: RuntimeValue = match (repeat, right) {
        (DiceCount::Int(i), DRightSide::List(list)) => {
            RuntimeValue::Pool(Rc::new(make_pool(i, list).map_err(negate)?))
        }
        (DiceCount::Int(i), DRightSide::Pool(p)) => {
            let mut new_pool = (*p).clone();
            if i < 0 {
                new_pool = new_pool
                    .try_map_outcomes(|outcome| outcome.checked_neg())
                    .map_err(negate)?;
            }
            new_pool.set_dimension(new_pool.dimension() * i.unsigned_abs());
            RuntimeValue::Pool(Rc::new(new_pool))
        }
        (DiceCount::Pool(left_p), right) => {
            let left_p = (*left_p).sum();
            let right = match right {
                DRightSide::List(list) => Pool::from_list(1, list),
                DRightSide::Pool(p) => sum_pool(&p).map_err(|error| {
                    error.into_error(range, "rolling a variable number of dice")
                })?,
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
                        .map_err(negate)?,
                )
            } else {
                None
            };
            // Each multiplier duplicates the right-hand pool and sums it, so a
            // face that cannot be added stops the walk at the first multiplier
            // that needs more than one die.
            RuntimeValue::Pool(Rc::new(
                left_p
                    .try_flat_map(|count| {
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
                        Ok(sum_pool(&dup_right)?.into())
                    })
                    .map_err(|error: PoolSumFailure| {
                        error.into_error(range, "rolling a variable number of dice")
                    })?,
            ))
        }
    };
    Ok(result)
}

fn make_pool(n: i32, sides: Vec<ElementValue>) -> Result<Pool<ElementValue>, ElementOpError> {
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
