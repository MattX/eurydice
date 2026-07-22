use eurydice_engine::{
    eval::{Evaluator, RuntimeValue, ScalarValue},
    grammar,
    output::OutputValue,
};
use malachite::Natural;

fn run(program: &str) -> Result<Vec<(RuntimeValue, String)>, String> {
    let statements = grammar::BodyParser::new()
        .parse(program)
        .map_err(|error| error.to_string())?;
    let mut evaluator = Evaluator::new();
    for statement in statements {
        evaluator
            .execute(&statement)
            .map_err(|error| error.to_string())?;
    }
    Ok(evaluator.take_outputs())
}

fn tuple_ints(value: &ScalarValue) -> Vec<i32> {
    let ScalarValue::Tuple(fields) = value else {
        panic!("expected tuple, got {value:?}");
    };
    fields
        .iter()
        .map(|field| field.as_int().expect("expected integer tuple field"))
        .collect()
}

fn tuple_distribution(value: &RuntimeValue) -> Vec<(Vec<i32>, Natural)> {
    let RuntimeValue::Pool(pool, _) = value else {
        panic!("expected tuple pool, got {value:?}");
    };
    pool.ordered_outcomes()
        .iter()
        .map(|(outcome, weight)| (tuple_ints(outcome), weight.clone()))
        .collect()
}

fn tuple_output_distribution(value: RuntimeValue) -> Vec<(Vec<i32>, f64)> {
    let OutputValue::TupleDistribution(distribution) = OutputValue::from(value) else {
        panic!("expected tuple output distribution");
    };
    distribution.probabilities
}

#[test]
fn constructs_and_projects_tuples() {
    let outputs = run(r#"
        output [tuple 1 2]
        output [tuple 1 2 3]
        output [tuple 1 2 3 4]
        output [element 2 of [tuple 10 20]]
        output #[tuple 1 2 3]
        "#)
    .unwrap();

    assert!(matches!(
        outputs[0].0,
        RuntimeValue::Scalar(ScalarValue::Tuple(_))
    ));
    assert!(matches!(
        outputs[1].0,
        RuntimeValue::Scalar(ScalarValue::Tuple(_))
    ));
    assert!(matches!(
        outputs[2].0,
        RuntimeValue::Scalar(ScalarValue::Tuple(_))
    ));
    assert_eq!(outputs[3].0, 20.into());
    assert_eq!(outputs[4].0, 3.into());
}

#[test]
fn tuple_constructor_lifts_over_distributions() {
    let outputs = run("A: d2 B: d2 output [tuple A B]").unwrap();
    assert_eq!(
        tuple_distribution(&outputs[0].0),
        vec![
            (vec![1, 1], 1u32.into()),
            (vec![1, 2], 1u32.into()),
            (vec![2, 1], 1u32.into()),
            (vec![2, 2], 1u32.into()),
        ]
    );
}

const RISK_PROGRAM: &str = r#"
    function: risk round ATTACK:s versus DEFENCE:s {
      COMPARISONS: [lowest of #ATTACK and #DEFENCE]
      ATTACKER_LOSSES: 0
      DEFENDER_LOSSES: 0
      loop I over {1..COMPARISONS} {
        if I@ATTACK > I@DEFENCE {
          DEFENDER_LOSSES: DEFENDER_LOSSES + 1
        }
        else {
          ATTACKER_LOSSES: ATTACKER_LOSSES + 1
        }
      }
      result: [tuple ATTACKER_LOSSES DEFENDER_LOSSES]
    }
    ROUND: [risk round 3d6 versus 2d6]
"#;

#[test]
fn risk_round_returns_joint_distribution() {
    let outputs = run(&format!("{RISK_PROGRAM} output ROUND")).unwrap();
    assert_eq!(
        tuple_distribution(&outputs[0].0),
        vec![
            (vec![0, 2], 2890u32.into()),
            (vec![1, 1], 2611u32.into()),
            (vec![2, 0], 2275u32.into()),
        ]
    );
}

#[test]
fn tuple_distribution_can_be_projected_through_scalar_parameter() {
    let outputs = run(&format!(
        r#"
        {RISK_PROGRAM}
        function: field I:n of VALUE:n {{ result: [element I of VALUE] }}
        output [field 1 of ROUND]
        output [field 2 of ROUND]
        "#
    ))
    .unwrap();

    let expected_first = [(0, 2890u32), (1, 2611), (2, 2275)];
    let expected_second = [(0, 2275u32), (1, 2611), (2, 2890)];
    for (value, expected) in outputs
        .iter()
        .map(|output| &output.0)
        .zip([expected_first.as_slice(), expected_second.as_slice()])
    {
        let RuntimeValue::Pool(pool, _) = value else {
            panic!("expected projected pool");
        };
        assert_eq!(
            pool.ordered_outcomes()
                .iter()
                .map(|(value, weight)| (value.as_int().unwrap(), u32::try_from(weight).unwrap()))
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn tuples_support_enum_fields() {
    let outputs = run(r#"
        enum: RESULT { MISS, HIT }
        T: [tuple 3 HIT]
        output T
        output [element 2 of T]
        "#)
    .unwrap();
    let RuntimeValue::Scalar(ScalarValue::Tuple(fields)) = &outputs[0].0 else {
        panic!("expected tuple");
    };
    assert!(matches!(fields[1], ScalarValue::Enum { value: 1, .. }));
    assert!(matches!(
        outputs[1].0,
        RuntimeValue::Scalar(ScalarValue::Enum { value: 1, .. })
    ));
}

#[test]
fn tuples_have_structural_equality_and_can_be_die_faces() {
    let outputs = run(r#"
        output [tuple 1 2] = [tuple 1 2]
        output [tuple 1 2] != [tuple 2 1]
        output d{[tuple 1 2], [tuple 3 4]}
        "#)
    .unwrap();

    assert_eq!(outputs[0].0, 1.into());
    assert_eq!(outputs[1].0, 1.into());
    assert_eq!(
        tuple_distribution(&outputs[2].0),
        vec![(vec![1, 2], 1u32.into()), (vec![3, 4], 1u32.into()),]
    );
}

#[test]
fn integer_tuples_support_vector_arithmetic() {
    let outputs = run(r#"
        output [tuple 1 2] + [tuple 3 4]
        output [tuple 10 20 30] - [tuple 1 2 3]
        output -[tuple 1 (-2) 3 0]
        output [tuple 2 (-3)] * 4
        output -2 * [tuple 2 (-3)]
        output [tuple 7 (-7)] / 3
        "#)
    .unwrap();

    let expected = [
        vec![4, 6],
        vec![9, 18, 27],
        vec![-1, 2, -3, 0],
        vec![8, -12],
        vec![-4, 6],
        vec![2, -2],
    ];
    for ((value, _), expected) in outputs.iter().zip(expected) {
        assert_eq!(
            tuple_ints(match value {
                RuntimeValue::Scalar(value) => value,
                _ => panic!("expected tuple scalar"),
            }),
            expected
        );
    }
}

#[test]
fn tuple_sequences_sum_during_arithmetic_and_scalar_coercion() {
    let outputs = run(r#"
        function: identity VALUE:n { result: VALUE }
        output {[tuple 1 2], [tuple 3 4]} + [tuple 10 20]
        output [identity {[tuple 1 2], [tuple 3 4]}]
        output {1, 2}@{[tuple 1 2], [tuple 3 4]}
        "#)
    .unwrap();

    assert_eq!(
        outputs
            .iter()
            .map(|(value, _)| match value {
                RuntimeValue::Scalar(value) => tuple_ints(value),
                RuntimeValue::Pool(_, _) => tuple_output_distribution(value.clone())[0].0.clone(),
                _ => panic!("expected summed tuple"),
            })
            .collect::<Vec<_>>(),
        [vec![14, 26], vec![4, 6], vec![4, 6]]
    );
}

#[test]
fn multidimensional_tuple_dice_sum_componentwise() {
    let outputs = run(r#"
        output 2d{[tuple 1 0], [tuple 0 1]}
        A: d{[tuple 1 0], [tuple 0 1]}
        B: d{[tuple 1 0], [tuple 0 1]}
        output A + B
        output 0d{[tuple 1 2], [tuple 3 4]}
        output -2d{[tuple 1 0], [tuple 0 1]}
        "#)
    .unwrap();

    let expected = vec![(vec![0, 2], 0.25), (vec![1, 1], 0.5), (vec![2, 0], 0.25)];
    assert_eq!(tuple_output_distribution(outputs[0].0.clone()), expected);
    assert_eq!(tuple_output_distribution(outputs[1].0.clone()), expected);
    assert_eq!(
        tuple_output_distribution(outputs[2].0.clone()),
        [(vec![0, 0], 1.0)]
    );
    assert_eq!(
        tuple_output_distribution(outputs[3].0.clone()),
        vec![
            (vec![-2, 0], 0.25),
            (vec![-1, -1], 0.5),
            (vec![0, -2], 0.25),
        ]
    );
}

#[test]
fn typed_empty_tuple_pool_sums_to_the_zero_tuple() {
    let outputs = run("output d{[tuple 5 6]:0}").unwrap();
    assert_eq!(
        tuple_output_distribution(outputs[0].0.clone()),
        [(vec![0, 0], 1.0)]
    );
}

#[test]
fn untyped_empty_dice_preserve_the_additive_identity_until_constrained() {
    let outputs = run(r#"
        EMPTY: 2d{}
        function: identity VALUE:n { result: VALUE }
        output EMPTY + EMPTY + [tuple 1 2]
        output [tuple 1 2] + EMPTY + EMPTY
        output [identity EMPTY] + [tuple 1 2]
        output EMPTY + EMPTY
        output 2d{} + 2d{[tuple (d2) (d2)]}
        output 2d{[tuple (d2) (d2)]}
        output 2d{[tuple 9 8]:0} + [tuple 1 2]
        "#)
    .unwrap();

    for index in 0..3 {
        assert_eq!(
            tuple_output_distribution(outputs[index].0.clone()),
            [(vec![1, 2], 1.0)]
        );
    }
    let OutputValue::Distribution(defaulted) = OutputValue::from(outputs[3].0.clone()) else {
        panic!("unconstrained identity should default to an integer output");
    };
    assert_eq!(defaulted.probabilities, [(0, 1.0)]);
    assert_eq!(
        tuple_output_distribution(outputs[4].0.clone()),
        tuple_output_distribution(outputs[5].0.clone())
    );
    assert_eq!(
        tuple_output_distribution(outputs[6].0.clone()),
        [(vec![1, 2], 1.0)]
    );
}

#[test]
fn pool_evaluated_functions_merge_identity_and_tuple_results() {
    let outputs = run(r#"
        function: identity VALUE:n { result: VALUE }
        function: maybe tuple CONDITION:n {
            if CONDITION { result: [tuple 1 2] }
            result: [identity 2d{}]
        }
        output [maybe tuple d{0, 1}]
        "#)
    .unwrap();

    assert_eq!(
        tuple_output_distribution(outputs[0].0.clone()),
        [(vec![0, 0], 0.5), (vec![1, 2], 0.5)]
    );
}

#[test]
fn tuple_scalars_and_lists_are_converted_to_distributions_in_the_engine() {
    let mut outputs =
        run("output [tuple 1 2] output {[tuple 1 2], [tuple 3 4], [tuple 1 2]}").unwrap();

    let OutputValue::TupleDistribution(scalar) = OutputValue::from(outputs.remove(0).0) else {
        panic!("expected tuple scalar distribution");
    };
    assert_eq!(scalar.probabilities, [(vec![1, 2], 1.0)]);

    let OutputValue::TupleDistribution(list) = OutputValue::from(outputs.remove(0).0) else {
        panic!("expected tuple list distribution");
    };
    assert_eq!(
        list.probabilities,
        [(vec![1, 2], 2.0 / 3.0), (vec![3, 4], 1.0 / 3.0)]
    );
}

#[test]
fn rejects_invalid_tuple_operations() {
    for program in [
        "output [tuple 1 [tuple 2 3]]",
        "output [element 0 of [tuple 1 2]]",
        "output [element -2147483648 of [tuple 1 2]]",
        "output [element 3 of [tuple 1 2]]",
        "output [tuple 1 2] + 3",
        "output 3 / [tuple 1 2]",
        "output [tuple 1 2] + [tuple 1 2 3]",
        "output [tuple 1 2] / 0",
        "output [tuple 2147483647 0] + [tuple 1 0]",
        "enum: RESULT { A } output [tuple 1 A] * 2",
        "output [tuple 1 2] = [tuple 1 2 3]",
        "output 0d6 + [tuple 1 2]",
        "output 2d{1:0} + [tuple 1 2]",
        "enum: RESULT { A, B } output 2d{[tuple 1 A], [tuple 2 B]}",
    ] {
        assert!(run(program).is_err(), "expected error for {program}");
    }
}
