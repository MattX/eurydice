use eurydice_engine::{
    eval::{Evaluator, RuntimeValue, ScalarValue},
    grammar,
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
fn rejects_invalid_tuple_operations() {
    for program in [
        "output [tuple 1 [tuple 2 3]]",
        "output [element 0 of [tuple 1 2]]",
        "output [element -2147483648 of [tuple 1 2]]",
        "output [element 3 of [tuple 1 2]]",
        "output [tuple 1 2] + [tuple 3 4]",
        "output [tuple 1 2] = [tuple 1 2 3]",
        "output 2d{[tuple 1 2], [tuple 3 4]}",
    ] {
        assert!(run(program).is_err(), "expected error for {program}");
    }
}
