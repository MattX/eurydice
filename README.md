# Eurydice

Eurydice (aka everydice) is a library and DSL to maniuplate dice rolls, or more generally probability distribution over integers and integer sequences.

## Examples

Simple query

```
max 3d20
```

Represents rolling 3 d20s and keeping the largest value.

The following program computes the probability of winning a Risk attack of 6 attacking and 4 defending armies.

```
def risk_attacker_win_chance(atk_armies: int, def_armies: int) -> dist[int]:
    if atk_armies = 1:
        0
    elif def_armies = 0:
        1
    else:
        attacker_die = d6 repeat min(3, attacker)
        defender_die = d6 repeat min(2, attacker)
        die_to_compare = min(2, 3, attacker)
        atk_wins = (attacker_die top die_to_compare) > (defender_die top die_to_compare)
        atk_win_count = sum atk_wins
        def_win_count = die_to_compare - atk_win_count
        risk_attacker_win_chance map [atk_win_count, def_win_count]

risk_attacker_win_chance(6, 4)
```

## Manual

### Syntax

The syntax aims to be Python-like, although greatly simplified.

### Types

To understand the language, it's maybe easiest to start with the types you can manipulate. There are 2 scalar types:

* `int`: Integers (32-bit signed). Example: `42` is an `int`.
* `dist`: Distributions, which are essentially a mapping from lists of integers to floats representing their probabilities. Distributions can be over sequences of a finite length, in which case their type is `dist[k]` for some `k`, or over arbitrary sequences, in which case they are `dist[?]`. Examples: `d6` is a `dist[1]`, `d{0, [0, 0]}` is a `dist[?]`.

There are two corresponding homogenous list types:

* `list[int]` is an arbitrary-length sequence of `int`s. Example: `3 repeat 2` is a `list[int]`.
* `list[dist[k]]` and `list[dist[?]]` are lists of distributions over sequences of size `k` and over arbitrary sequences, respectively. Examples: `2d6` is a `list[dist[6]]`, and `[d6, d20]` is a `list[dist[?]]`.

There are two compound types:

* `tuple[type1, type2, ...]` is a fixed-length tuple of arbitrary types.
* `function<k, j,...>[arg1, arg2, ..., argn return]` is a function from `args1, ..., argn` to `return`.

#### Type casts and least upper bounds

All container types are covariant, so since `int :< dist[1]`, `list[int] :< list[dist[1]]`.

* `int` unifies with `dist[1]`, creating a constant distribution.
* `dist[k]` for any `k` unifies with `dist[?]`
* `tuple[t1, t2, ...]` where `t1 = t2 = ...` unifies with `list[t1]`.
* No other types unify

#### Primitive types

`applyd: function<k, j>[dist[k], tuple[int, k] -> tuple[int, j]] -> dist[j]`
`sum: function<k>[list[dist[k]]] -> dist[k]`

## To do list

(Besides immediate concerns)

* **Move to log-probability**
* `var`s as a third type?
* generics over `k` in `dist[k]`.
