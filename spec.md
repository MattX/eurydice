# Eurydice language specification

This specification defines how Eurydice should behave. Any divergence between Eurydice and this description is a bug; please report it.

This document is intended primarily for other AnyDice implementers and power users. If you're looking for an introduction to the AnyDice DSL, this document is likely too technical. See the [AnyDice docs](https://anydice.com/docs/) for a gentler introduction.

**This project is not affiliated with AnyDice or Jasper Flick.**

## Overview

Eurydice is a dialect of the [AnyDice](https://anydice.com/) language. It is almost completely compatible with AnyDice, with some extensions, and a few restrictions.

> [!IMPORTANT]
> Intentional divergences from AnyDice are marked with information boxes like this one. Sections marked ⊕ describe Eurydice-only features.

Eurydice is reproducible and hermetic: programs can't interact with the environment, and two runs will have exactly the same results.

This document frequently uses the following notations to describe pools:

* `<n>d<k>` represents a pool of `n` dice with `k` faces labeled 1 through `k`.
* `<n>d{<k1>:<count1>,...,<kn>:<countn>}` represents a pool of `n` dice with `count1` faces showing `k1`, `count2` faces showing `k2`, and so on. If `:<count>` is omitted for any `k`, the count defaults to 1.

This is exactly compatible with Eurydice expressions using the `d` operator.

## Lexical elements

A Eurydice program is composed of a sequence of statements. Eurydice is whitespace-insensitive, except to separate tokens.

_Block comments_ start and end with backslashes: `\ this is a comment \`.

_Line comments_ start with three backslashes: `\\\ this is a line comment`.

> [!IMPORTANT]
> AnyDice does not support line comments.

Comments are treated as whitespace, and can be used to separate tokens, e.g. `[hello\a comment\world]` is equivalent to `[hello world]`.

_Variable identifiers_ are composed of uppercase letters and underscores, for instance `X`, `MY_VARIABLE`, or `_MY_VARIABLE`. `MyVariable` and `x` are not variable identifiers.

_Words_ are composed of lowercase letters and underscores (after the first position only), except the single lowercase letter `d`, which is punctuation rather than a word. `word`, `if`, `output`, and `my_word` are all words. `WORD`, `wOrd`, and `_word` are not.

> [!IMPORTANT]
> AnyDice words (function names) can't contain underscores.

_Punctuation_: the following characters or sequences of characters are tokens:

```
+   =   &   @   :   (   ..
-   !=  |   d   [   )   ,
*   <   !   #   ]   {
/   <=              }
^   >
    >=
```

_Integer literals_: Integer literals are base-10 representations of unsigned 31-bit integers. It is an error if an integer literal represents a number greater than 2^31-1 (2147483647). These literals represent a subset of the underlying `int` type, which holds 32-bit signed integers; negative values can be obtained with the unary minus operator. Examples include `0` and `123`.

_String literals_: String literals can occur only in a few places. They are enclosed in double quotes (`"`). A string literal can contain escaped double quotes: `"message: \"hello\""`. No other escape sequences are supported.

## Program execution

A Eurydice program is composed of a sequence of statements, which are executed in order. Once the program has finished executing, all outputs are displayed to the user.

Errors are fatal and terminate program execution.

## Data types

The type grammar for Eurydice is

```
Scalar  = int | E
          where E ranges over declared enums
Element = Scalar | tuple of 2 to 4 Scalars
Type    = Element | list [Element] | pool [Element]
```

There are three kinds of element values:

* `int`: values of this type hold a 32-bit signed integer
* _enum members_: values of a user-declared type which contains a set of nominal, non-numeric values, which can be used to represent discrete outcomes, booleans, etc.
* _tuples_: fixed-size, ordered products of two to four `int` or enum member fields. Tuple types are structural: the type of a tuple is determined by the number and types of its fields. Tuples do not nest.

Two kinds of collections are built on these:

* _lists_: values of this type hold a list of one particular element type.
* _pools_: values of this type hold a pool of one element type `e`. A pool consists of a mapping from outcomes of type `e` to probabilities (whose representation is unspecified), together with an unsigned count of dice called the _dimension_.

Lists and pools are homogeneous and cannot mix values with different outcome types. For example, all tuples in one list must have the same arity and corresponding field types.

The outcome type of a collection is bracketed in the grammar because it may be _absent_: an empty collection that carries no information about what it could contain has no outcome type, and adopts one from its context. Only collections can lack an outcome type; there is no element value whose type is absent. See [empty collections](#-empty-collections) for discussion.

The maximum number of elements in a list, or of outcomes in a pool, is 2^31-1.

There are no first-class functions.

> [!IMPORTANT]
> A reader concerned only with AnyDice compatibility may read `Element` as `int` throughout, treat every ✗ in the table below as "does not arise", and skip those sections entirely.
>
> AnyDice does not restrict integers to 32-bit values. Experimentation suggests that AnyDice integers are represented as double-precision floats (`output 9007199254740993` returns 9007199254740992).

### What element types support

| | `int` | enum member | tuple of `int`s | tuple with an enum field |
|---|---|---|---|---|
| `=`, `!=` | ✓ | same enum only | same tuple type | same tuple type |
| `<`, `<=`, `>`, `>=` | ✓ | ✗ | ✗ | ✗ |
| `+`, `-`, unary `-` | ✓ | ✗ | componentwise | ✗ |
| `*`, `/` | ✓ | ✗ | by an `int` only | ✗ |
| `^`, `&`, `\|`, `!` | ✓ | ✗ | ✗ | ✗ |
| `#` | digit count | `1` | arity | arity |
| endpoint of a `..` range | ✓ | ✗ | ✗ | ✗ |
| dice count or side count for `d`, index for `@` | ✓ | ✗ | ✗ | ✗ |
| ordering: `[sort]`, `"position order"` | ✓ | ✗ | ✗ | ✗ |
| summing a pool of dimension > 1 | ✓ | ✗ | ✓ | ✗ |

**Every operation requires `int` operands unless this document says otherwise.** Supplying an enum member or a tuple anywhere else is an error. Besides the ✓ entries above, the operations that accept other element types are collection construction, `@` selecting a single element, [`[SEQ contains N]`](#seqs-contains-nn), [`[count NEEDLES in HAYSTACK]`](#count-needless-in-haystacks), the [tuple built-ins](#-tuples), and `print`/`output`.

An element type is _additive_ when it has a ✓ in the `+` row: `int`, and tuples all of whose fields are `int`. A list or pool is additive when its outcome type is. Only additive values can be summed, and only additive collections can be summed to an element.

> [!IMPORTANT]
> In AnyDice the only element type is `int`, so every column but the first is empty, everything is additive, and the rule above imposes no restrictions.

### ⊕ Enums

Enums must be declared at the top level:

```
enum: ATTACK_RESULT { MISS, HIT, CRITICALHIT }
```

The enum type and its members use variable-identifier syntax. An enum must contain at least one member. Variables, enum type names, and enum member names all share a namespace. It is an error to declare an enum whose name or any of whose member names equals that of a declared variable. Conversely, it is an error to declare a variable whose name collides with an enum name or member name.

Enum members support equality and inequality with members of the same enum, as set out in [What element types support](#what-element-types-support).

### ⊕ Tuples

Tuples are constructed by the built-in functions `[tuple A B]`, `[tuple A B C]`, and `[tuple A B C D]`. Each field must be a `Scalar`. Nested tuples are not supported.

```
PAIR: [tuple 3 5]
output [field 1 of PAIR]  \ Outputs 3
output [field 2 of PAIR]  \ Outputs 5
output #PAIR              \ Outputs 2
```

Tuple fields are selected with `[field INDEX of TUPLE]`. Indices are one-based and must be within the tuple's bounds; an invalid index is an error. The `@` operator does not project tuple fields.

Tuples support structural equality and inequality with tuples of the same type: two tuples are equal exactly when every pair of corresponding fields is equal. An _additive_ tuple, one whose fields are all of type `int`, additionally supports arithmetic: `+`, `-`, and unary `-` operate componentwise, multiplication by an `int` is supported in either operand order, and division by an `int` operates componentwise. `int / tuple` is not defined. Everything else follows the default in [What element types support](#what-element-types-support).

Tuple constructors and field projection participate in normal [pool-based function evaluation](#pool-based-evaluation). Consequently, constructing a tuple from pool-valued arguments produces their joint distribution, and passing a tuple-valued pool to an `n` parameter evaluates the function once per tuple outcome.

```
A: d2
B: d2
JOINT: [tuple A B]
\ JOINT has outcomes (1, 1), (1, 2), (2, 1), and (2, 2). \
output [field 1 of JOINT]
```

### Pools

It is not possible to create a pool value representing a pool of different types of dice. For instance, `3d6` is a pool of three d6s, but there is no way to represent a pool of one d6 and one d8.

Pools whose outcomes are not additive (essentially, those containing enums) must have dimension one. A pool with no outcome type may have any dimension.

Pools may have no possible outcomes (for instance, such a pool is created with expression `d{}`). Such a pool has dimension 0.

The following operations are frequently referred to in this document:

#### Summing

_Summing_ an [additive pool](#what-element-types-support) transforms it into a pool of dimension 1 whose outcomes are the possible sums when sampling from the dice in the pool, with associated probabilities. Integer outcomes are added normally and additive tuple outcomes are added componentwise.

For instance, summing `2d2` results in a pool equivalent to `d{2, 3, 3, 4}`.

Summing a pool of dimension 1 does nothing, including for a non-additive pool. Summing a dimension-0 pool whose outcome type is known creates a dimension-1 pool containing that type's zero: `0` for integers, or an all-zero tuple of the appropriate type. Summing a pool with no outcome type is covered in [Empty collections](#-empty-collections).

#### Multiset iteration

_Iterating over the multisets_ in a pool refers to processing each possible _sorted_ outcome of the collection of dice in the pool, together with its associated probability. For instance, the multisets of `3d{1, 2:2, 3:3}` are:

```
multiset              probability
{1, 1, 1}:   (1/6)^3                     * 1
{1, 1, 2}:   (1/6)^2 *  1/3              * 3
{1, 1, 3}:   (1/6)^2           *  1/2    * 3
{1, 2, 2}:    1/6    * (1/3)^2           * 3
{1, 2, 3}:    1/6    *  1/3    *  1/2    * 6
{1, 3, 3}:    1/6              * (1/2)^2 * 3
{2, 2, 2}:             (1/3)^3           * 1
{2, 2, 3}:             (1/3)^2 *  1/2    * 3
{2, 3, 3}:              1/3    * (1/2)^2 * 3
{3, 3, 3}:                       (1/2)^3 * 1
```

The probability associated with each multiset is the product of its values' probabilities, multiplied by the number of orderings that produce it. That count is the multinomial coefficient $\frac{n!}{m_1!\,m_2!\cdots m_k!}$, where $n$ is the number of dice in the pool and $m_i$ is the number of times the $i$-th distinct value occurs in the multiset. These ten probabilities sum to 1.

If there is a single die in the pool, this is the same as iterating over the die's outcomes.

#### Outcome mapping

_Outcome mapping_ a pool transforms each of its outcomes with a specified operation. If two outcomes are transformed into the same value, their probabilities are summed. The operation applies to individual outcomes, not to multisets, so a pool of dimension greater than one is [summed](#summing) first by whichever operation calls for the mapping.

For instance, outcome mapping `2d{-1..1}` with the operation `x→2*abs(x)` results in a pool equivalent to `d{0:3, 2:4, 4:2}`.

#### Flat mapping

_Flat mapping_ a pool uses a function that turns a multiset into a pool. The function is called for each multiset of the original pool. The resulting pools are combined according to the probability of each original multiset.

Each produced pool is summed if its dimension is not 1. The resulting pool always has a dimension of 1.

In pseudocode:

```
def flat_map(pool, f):
    final_outcomes = {}
    for (outcome, prob) in pool.multisets():
        for (new_outcome, new_prob) in f(outcome).sum():
            final_outcomes[new_outcome] += prob * new_prob
    return final_outcomes
```

For instance, flat mapping `d3` with an operation `x→d(x)` produces a pool equivalent to `d{1:11, 2:5, 3:2}`. `d3`'s outcomes are 1, 2, and 3, with equal probabilities of 1/3.

* 1 maps to `d1` = `d{1}`
* 2 maps to `d2` = `d{1, 2}`
* 3 maps to `d3` = `d{1, 2, 3}`

In the final result, 1 gets probability 1/3 \* 1 + 1/3 \* 1/2 + 1/3 \* 1/3 = 11/18; 2 gets probability 1/3 \* 1/2 + 1/3 \* 1/3 = 5/18; and 3 gets probability 1/3 \* 1/3 = 2/18, equivalent to `d{1:11, 2:5, 3:2}`.

#### Multiset cross product

Starting with pools $p_1, p_2, \ldots, p_n$, the _multiset cross product_ is an iterator over pairs of (list of multisets, probability), where:

* The lists range over all possible combinations containing one multiset from each pool's multiset iterator.
* The probability associated with each list is the product of the multisets' probabilities.

In pseudocode:

```
for (multiset_1, prob_1) in p_1.multisets():
    ...
        for (multiset_n, prob_n) in p_n.multisets():
            yield ([multiset_1, ..., multiset_n], prob_1 * ... * prob_n)
```

### ⊕ Empty collections

Everything in this section concerns collections that are empty and carry no information about what they could have contained. In AnyDice, where the only element type is `int`, none of it is observable: an empty collection sums to `0` and that is the end of it.

#### Absent outcome types

A bare empty list literal `{}`, and the empty pool `d{}`, have no outcome type. Because they contain no values, they are compatible with any outcome type their context supplies, so `{{}, {MISS}}` is valid and produces a list of enum members. An explicitly typed empty literal such as `{1:0}` instead retains the outcome type of its repeated expression, so `{{1:0}, {MISS}}` is an error.

This gives a way to write an empty collection of a chosen type, either from a sample element or from an existing collection:

```
EMPTY_PAIRS: {[tuple 0 0]:0}   \ empty, with outcome type tuple(int, int) \

function: drop all SEQ:s {
  result: {SEQ:0}              \ empty, with SEQ's outcome type \
}
```

#### The empty sum

Summing a collection with no outcome type produces _the empty sum_, written `𝑒`. It is a single value, not a type the user can write.

`𝑒` adopts the type of whatever it is combined with: added to an `int` it is `0`, and added to an additive tuple it is that tuple's all-zero value. Where an `int` is required and `𝑒` is still unconstrained, it becomes the integer `0`; this is what makes `{} + 1` evaluate to `1`, as in AnyDice. The `print` statement displays an unconstrained `𝑒` as italic `𝑒`; `output` displays it as `0`.

Summing a pool with no outcome type, of any dimension, produces a pool of dimension 1 with no outcomes, whose outcome type is that of `𝑒`. This is what keeps `d{} + 2d{} + d{[tuple 1 2]}` well-typed.

#### Why this exists

Without `𝑒`, a recursion that accumulates tuples could not terminate through the [`maximum function depth`](#global-settings) setting, because the empty list returned by the truncated call would sum to the integer `0` and adding it to a tuple would be an error. That setting is the only termination mechanism available to hand-rolled exploding and rerolling dice, which is the idiom this rule exists to support:

```
function: attack DIE:n {
  if DIE = 6 { result: [tuple 6 1] + [attack d6] }
  result: [tuple DIE 0]
}
output [attack d6]
```

This yields the joint distribution of total damage and number of explosions. The recursion is bounded only by `maximum function depth`; the truncated call sums to `𝑒`, leaving the accumulated tuple untouched.

## Values, variables, and bindings

All Eurydice _values_ are immutable: it is not possible to modify a list or a pool in place.

Eurydice uses [dynamic scoping](<https://en.wikipedia.org/wiki/Scope_(computer_science)#Lexical_scope_vs._dynamic_scope>). At every point during execution, an _environment_ maps variable names to values and function names to functions. Outside a function, the environment is known as the _global environment_. Before execution enters a function's body, a new _environment frame_ is pushed onto the stack of environments. This frame is popped and deleted when execution leaves the function.

```
function: dynamic one {
    X: 1
    result: [x]
}

function: dynamic two {
    X: 2
    result: [x]
}

function: x {
    result: X
}

output [dynamic one]  \\\ Outputs 1
output [dynamic two]  \\\ Outputs 2
```

Values can be bound to variable names in three ways: through an assignment statement, through function parameters, or through a `loop` statement. All methods only affect the innermost environment frame.

Functions can be bound to function names in just one way: with a `function` statement. This method also only affects the innermost frame.

Variable and function bindings are mutable: new values can be assigned to variables, and new functions can be assigned to function names. Only bindings in the innermost environment frame can be mutated. Reusing a name from an outer frame in an assignment will _shadow_ the outer binding while the inner frame is live.

```
function: x {
    result: X
}

function: reassign x {
    X: 2
    result: X
}

X: 0
output [x]           \\\ Outputs 0
X: 1
output [x]           \\\ Outputs 1
output [reassign x]  \\\ Outputs 2
output [x]           \\\ Outputs 1 - the global environment
                     \\\ was not modified by [reassign x].
```

## Global settings

There are three named global settings:

* `explode depth`: a non-negative integer with a default of 2.
* `maximum function depth`: a non-negative integer with a default of 10. If the current recursion depth already equals this setting, an attempted function call does not execute and instead returns an empty list with [no outcome type](#-empty-collections). Summing that list yields the [empty sum](#the-empty-sum), allowing truncated recursion to keep accumulating values of any additive type.
* `position order`: either `"lowest first"` or `"highest first"` (the default). This setting affects three things:
  * the behavior of the [`@` operator](#-operator),
  * the behavior of the [sort function](#sort-sequences),
  * the behavior of [calling a function over pools](#pool-based-evaluation)

Settings can only be set outside of a function, using the [`set` statement](#set).

## Expressions

### References

The name of a variable is an expression. It evaluates to the value of that variable in the innermost environment frame that has a binding for that variable name.

It is an error to refer to an unbound variable name.

```
X: 3
output X  \\\ Outputs 3
X: {1, 2}
output X  \\\ Outputs {1, 2}
X: d4
output X  \\\ Outputs d4
```

```
output Y  \\\ Error
```

Note that assigning a pool to a variable _does not_ attach the variable to a particular outcome. For instance,

```
X: d4
output X + X   \\\ Output outcomes are {2, 3, 4, 5, 6, 7, 8}
               \\\ *not* {2, 4, 6, 8}.
```

To achieve this effect, use a [pool-evaluated function](#pool-based-evaluation):

```
function: add D:n to self { result: D + D }
output [add d4 to self]  \\\ Output outcomes: {2, 4, 6, 8}
```

### Literals

The value of an integer literal is the integer it represents.

```
output 25  \\\ Outputs 25
```

### List expressions

A list expression is a comma-separated list of _list elements_, enclosed in curly braces:

```
ListExpr = '{' {ListElem ','} [ListElem [',']] '}'.
```

Once all list elements are evaluated, they are each [flattened into a list](#list-flattening), and concatenated.

#### List elements

Each list element is one of an expression or a range, optionally followed by a repeat count:

```
ListElem = (Expr | Expr '..' Expr) [':' Expr].
```

Expressions composing a list element are evaluated in an unspecified order. If the list element is a range, both expressions must evaluate to `int` values, and the range is replaced by a list containing all integers between the start and end of the range, inclusive. If the start of the range is greater than the end, an empty list is produced. Because a range can only produce integers, that list has outcome type `int`, unlike the bare literal `{}`, which has [no outcome type](#-empty-collections).

If a repeat expression is present, it must evaluate to an `int`. If this value is negative, it is replaced with `0`. The primary expression is then [flattened into a list](#list-flattening), and is concatenated to itself the indicated number of times. If the repeat count is 0, the empty list is produced.

#### List flattening

List flattening transforms values in the following way:

* Element values are flattened by enclosing them in a single-element list.
* `list` values are unchanged by flattening.
* `pool` values are flattened first by [summing](#summing), then by discarding the probabilities and creating a list containing each outcome in ascending order. Outcomes with nonzero probability appear once regardless of their probabilities.

Flattening preserves `pool` outcome types. Non-additive `pool`s always have dimension one, so flattening them does not require adding their outcomes together.

#### Examples

```
output {}                \\\ Outputs {}
output {1, 2, 3}         \\\ Outputs {1, 2, 3}
output {d4}              \\\ Outputs {1, 2, 3, 4}
output {d2:2}            \\\ Outputs {1, 2, 1, 2}
output {{1, 2}, {3, 4}}  \\\ Outputs {1, 2, 3, 4}
```

### Parenthesized expressions

```
'(' Expr ')'
```

A single expression enclosed in parentheses yields the result of that expression. Explicit parentheses may be used for clarity, or to override the default association of subexpressions.

### Unary operators

There are four unary operators, all of which bind more tightly than any binary operator: `!`, `-`, `#`, and `d`.

`!` and `-` compute logical and arithmetic negation respectively. `!` requires an `int`. `-` accepts all additive values; it negates tuple fields componentwise.

* If the argument is an `int`, `-` negates the value, while `!` evaluates to 0 if the argument is nonzero and 1 otherwise.
* If the argument is a `list`, its values are summed to an element, to which the operator is applied.
* If the argument is a `pool`, the pool is summed, then the [outcomes of the pool are all mapped](#outcome-mapping) with the operator.

`#` evaluates to the length of its argument:

* If the argument is an `int`, it returns the number of digits in the base-10 representation. `#0` evaluates to `1`. A leading `-` is not counted as a digit.
* If the argument is an enum member, it returns `1`.
* If the argument is a tuple, it returns the tuple's arity.
* If the argument is a `list`, it returns the number of elements.
* If the argument is a `pool`, it returns the pool's dimension (count of dice in the pool).

The `d` unary operator acts as the `d` binary operator, with a left argument of 1.

### Binary operators

This is the binary operator precedence list for Eurydice, in decreasing order of precedence:

1. `d`
2. `@`
3. `^`
4. `*`, `/`
5. `+`, `-`
6. `=`, `!=`, `<`, `<=`, `>`, `>=`
7. `&`, `|`

All operators are left-associative.

#### `d` operator

The `d` operator is the main way to create a pool.

1. The RHS operand is converted to a pool with the following operation:
   1. If it is an `int` `i`, it is converted to a pool with values 1 to `abs(i)` inclusive (if `i` is 0, the pool contains the single outcome 0). If `i` is negative, the pool outcomes then are mapped to their opposite. For instance, `d(-3)` evaluates to `d{-1, -2, -3}`.
   2. If it is a `list`, it is converted to a pool whose outcomes are the distinct values in the list, and whose probability for each outcome is proportional to the number of occurrences of each value in the list. An empty list produces a pool of dimension 0.
   3. Pools provided as an RHS operand are not transformed.
2. The LHS operand is first summed if it is a `list` (which must be an additive list), then:
   1. If it is an `int` `i`, the dimension of the RHS pool is multiplied by `abs(i)`. If `i` is negative, then each outcome in the resulting pool is multiplied by `-1`.
   2. If it is a pool with `int` outcomes, then the LHS is [flat mapped](#flat-mapping) with the operation described in step 1.1. (Recall that flat mapping takes a pool and an `int -> pool` function; the operation described in step 1.1 is such a function.)
   3. It is an error if a value of another type is used as LHS to `d`.

#### `@` operator

The `@` operator selects the (LHS)-th element from its RHS. A tuple is not treated as a list by this operator; use [`[field INDEX of TUPLE]`](#-field-indexn-of-tuplen) to project a tuple field.

> [!NOTE]
> `@` deliberately does not project tuple fields, because on a pool of tuples the two meanings would collide silently. `1@POOL` already means "select the first die of each multiset", which for a pool of tuples yields a whole tuple; projection would instead yield one field of every outcome. Both readings are well-typed and return plausible values, so overloading `@` would turn a wrong choice into a wrong number rather than an error. Field projection is therefore always spelled with the `[field ... of ...]` built-in, whose behavior on a pool is fixed by [pool-based evaluation](#pool-based-evaluation).

The LHS supplies the indices, and must be an `int` or a list of `int`s; anything else is an error. An `int` LHS is converted to a singleton list.

* If the RHS is an `int` or `list`, for each value `i` in the resulting LHS list:
  * If the RHS is an `int` `j`, the base-10 digit of `abs(j)` at index `i` is selected. If `j` is negative, this digit is multiplied by -1. If the `"position order"` [global setting](#global-settings) is set to `"highest first"`, index 1 corresponds to the most significant digit; otherwise, to the least significant. Valid indices start at 1.
  * If the RHS is a `list`, its element at position `i` is selected. The first element of the list has index 1. This is not affected by the `"position order"` setting.
  * In both the int and list RHS cases, if `i` is invalid (zero or negative, or greater than list length or digit count), the expression evaluates to 0.
  * Finally, all selected elements are summed. Selecting a single element works for any RHS outcome type; it is an error to select a number of elements other than one from a RHS whose outcome type is not additive, since they could not be summed.
* If the RHS argument is a `pool`, each outcome multiset is flat mapped with the following function:
  * The multiset is sorted according to the `"position order"` [global setting](#global-settings).
  * Elements are selected from the multiset and summed as if using the `@` operator from a list.

#### Mathematical operators

The operators `^`, `*`, `/`, `+`, `-`, `&`, and `|` are mathematical operators, performing exponentiation, multiplication, integer division, addition, subtraction, logical conjunction, and logical disjunction, respectively.

Applied to `int`s, `&` evaluates to `1` if both of its arguments are nonzero, `0` otherwise; `|` evaluates to `1` if at least one argument is nonzero. Other mathematical operators have standard behavior on the integers.

Division by 0 causes an error to be raised. `0^0` evaluates to 1.

Operators have the following signatures on element values:

```
^ * / + - & | : int, int                       -> int
+ -           : additive tuple, additive tuple -> additive tuple
*             : int, additive tuple            -> additive tuple
* /           : additive tuple, int            -> additive tuple
```

The [empty sum](#the-empty-sum) `𝑒` may stand in for either operand of `+`, `-`, `*` and `/`, adopting the other operand's type; `𝑒` combined with `𝑒` is again `𝑒`.

If the arguments are not both elements, the coercion rules are the following:

1. It is an error if either argument is a non-additive list or pool.
2. If either argument to a mathematical operator is an additive `list`, it is summed to an element.
3. After this, if both arguments are elements, the operator is applied according to the rules above.
4. Otherwise, both arguments are converted to `pool`s and summed. The LHS is then flat mapped with a function that applies the operator to the LHS and each value in the RHS.

#### Comparison operators

The operators `=`, `!=`, `<`, `<=`, `>`, and `>=` are comparison operators, performing equality, inequality, less-than, less-than-or-equal, greater-than, and greater-than-or-equal checks, respectively.

In all cases, these operators evaluate to `1` if their condition is true, or `0` otherwise.

Equality and inequality are also defined nominally for enum members and structurally for tuples of the same type. Comparing values with different types is an error.

1. If both arguments are elements, equality or inequality compares them directly. Ordering comparisons require two `int`s.
2. If one argument is a `list`, and the other is an element of the same outcome type, the comparison is performed between the element and each member of the `list`. The expression's value is an `int`: the count of comparisons that evaluated to true. Ordering in this form requires `int` outcomes.
3. If both arguments are `list`s, equality and inequality compare the complete lists structurally. Lists of `int`s may additionally be compared in [lexicographic order](https://en.wikipedia.org/wiki/Lexicographic_order).
4. If either argument is a `pool`, both arguments are converted to `pool`s and summed. The LHS is then flat mapped with a function that applies the operator to the LHS and each value in the RHS. Additive tuple pools are summed componentwise. For non-additive pools, only equality and inequality are available, and summing their dimension-one pools is a no-op.

### Function calls

Functions can be called by enclosing the words that make up their names, interleaved with expressions, between square brackets:

```
ArgumentTail = SafeExpr | ',' Expr.
FunctionCallItem = Word | Expr {ArgumentTail} Word.
FinalFunctionCallItem = Expr {ArgumentTail}.
FunctionCall = '[' {FunctionCallItem} [FinalFunctionCallItem] ']'.
```

A technical trick is needed in the grammar because arguments are allowed to follow each other with no word in between: `SafeExpr` is any expression that does not start with the tokens `-` or `d`. Because expressions can immediately follow each other, it would otherwise not be possible to know if a call like `[f 1 -2]` should be parsed as `[f (1) (-2)]` or as `[f (1-2)]`. An `Expr` in the function call must therefore be followed by a `Word`, which clearly separates the two arguments, or by a `SafeExpr`, which cannot be a suffix of an `Expr` and is thus necessarily a new argument.

A comma may also be used to separate two adjacent arguments. Because the comma has already done the separating, any `Expr` may follow one, including expressions starting with `-` or `d`:

```
output [tuple d6, d8]   \\\ Two arguments
output [tuple d6 d8]    \\\ Error: one argument, the expression d(6d8)
```

Commas are pure separators: they are not part of the function's identifier. `[add 1, 2]` and `[add 1 2]` resolve to the same function. A comma is only valid between two arguments, so a leading, trailing, or doubled comma is an error, as is a comma before a word.

> [!IMPORTANT]
> AnyDice does not accept commas between arguments. Code using them is not portable.

To evaluate a function call, the function identifier is resolved first. A function is identified by the sequence of words and argument locations in its name, and it is an error if the function's identifier is not in scope in the current environment. Here are some examples:

```
function: add A:n B:n { result: A + B }
output [add 1 2]    \\\ outputs 3
output [add 1, 2]   \\\ outputs 3 - same function
output [add 1 - 2]  \\\ error: one argument, the expression 1-2

function: add A:n and B:n { result: A + B }
output [add 1 and 2]   \\\ outputs 3
output [add 1 and -2]  \\\ outputs -1
```

After the identifier has been resolved, function calls proceed in several steps.

#### Argument coercion

The actual types of the arguments are compared to the expected argument types, and each is transformed in the following way:

* If the expected argument type is unspecified, or if the expected shape and outcome type match the actual value, the value is not transformed.
* If the actual argument is a `list`:
  * If an element is requested, an additive `list` is summed to an element. It is an error if the list is not additive.
  * If a `pool` is requested, a `pool` is created, with equally likely outcomes from the `list`.
* If the actual argument is an element:
  * If a `list` is requested, a singleton list is created.
  * If a `pool` is requested, a single-outcome `pool` is created.
* If the actual argument is a `pool` and an element is requested, the pool is summed, creating a new `pool` (see the note below: `pool`-typed values can be passed to element-typed arguments).

After this process, some values of type `pool` may still correspond to arguments where element or `list` shapes are requested. If this is not the case, the function is called once, and the value of the expression is the result of [evaluating the function](#function-evaluation). If it is the case, evaluation proceeds as described in the next section.

#### Pool-based evaluation

This section applies if any `pool` values are passed to element- or `list`-typed arguments. As mentioned above, any pool corresponding to an element argument is summed to dimension 1. Then, the [multiset cross product](#multiset-cross-product) of the pools is generated. The order in which elements inside each multiset are ordered follows the `"position order"` [global setting](#global-settings).

The function is then evaluated once for each value in the multiset cross product (this may be 0 times if the cross product is empty). In each invocation, the argument values are:

* For any argument whose actual shape matches its declared shape, the value is passed untransformed.
* For arguments where a `pool` was provided but a `list` or element was requested, that pool's multiset value in the current item of the multiset cross product iterator. An element argument receives the multiset's sole value, which may be of any element type.

Each invocation's result is converted to a `pool` through the standard conversions. The final result of the function call is a `pool` created by iterating through every possible outcome of the intermediate pools, and summing the probability of that outcome in each of the intermediate pools, multiplied by the probability of the value that generated this intermediate pool.

For instance, suppose the following call needs to be evaluated:

```
function: one or N:n th die from FACES:s {
  result: d{1, N@FACES}
}
output [one or d3 th die from 3d2]
```

The cross-product multisets and their corresponding intermediate pools are shown below. `"position order"` has its default value of `"highest first"`, so each `FACES` multiset is ordered from its highest value down, and `N@FACES` indexes into it in that order.

| Multiset for `N` | Multiset for `FACES` | Weight | Outcome | Weight for `1` | Weight for `2` |
|------------------|----------------------|--------|---------|----------------|----------------|
| [1]              | [1, 1, 1]            | 1      | d{1}    | 1              |                |
| [2]              | [1, 1, 1]            | 1      | d{1}    | 1              |                |
| [3]              | [1, 1, 1]            | 1      | d{1}    | 1              |                |
| [1]              | [2, 1, 1]            | 3      | d{1, 2} | 3/2            | 3/2            |
| [2]              | [2, 1, 1]            | 3      | d{1}    | 3              |                |
| [3]              | [2, 1, 1]            | 3      | d{1}    | 3              |                |
| [1]              | [2, 2, 1]            | 3      | d{1, 2} | 3/2            | 3/2            |
| [2]              | [2, 2, 1]            | 3      | d{1, 2} | 3/2            | 3/2            |
| [3]              | [2, 2, 1]            | 3      | d{1}    | 3              |                |
| [1]              | [2, 2, 2]            | 1      | d{1, 2} | 1/2            | 1/2            |
| [2]              | [2, 2, 2]            | 1      | d{1, 2} | 1/2            | 1/2            |
| [3]              | [2, 2, 2]            | 1      | d{1, 2} | 1/2            | 1/2            |

The final result has outcome 1 with weight 18, and outcome 2 with weight 6. (Setting `"position order"` to `"lowest first"` reverses which value of `N` selects which face, but produces the same totals here, because `N` is uniform.)

#### Function evaluation

A [new environment frame](#values-variables-and-bindings) is created, binding the value of each expression to its corresponding argument name. Each statement in the function's body is then evaluated sequentially. If a [return statement](#return-from-function) is encountered, execution returns to the caller, and the function call's value is the return statement's expression.

If no return statement is encountered, the function returns a pool of dimension 1 with no outcomes and no outcome type. Unlike an empty list, this contributes no outcomes to the caller rather than summing to zero:

```
function: sometimes nothing A:n {
  if A >= 5 { result: A }
}
output [sometimes nothing d10]  \\\ Outcomes 5 through 10 only

function: sometimes empty A:n {
  if A < 5 { result: {} }
  result: {A}
}
output [sometimes empty d10]    \\\ Outcome 0 with probability 40%, then 5 through 10
```

## Statements

```
Statement = IfStatement | LoopStatement | PrintStatement | OutputStatement
          | FunctionDefinitionStatement | ResultStatement | AssignmentStatement
          | EnumDefinitionStatement | SetStatement.
Block = '{' {Statement} '}'.
```

### Conditionals

Statements can be executed conditionally:

```
IfStatement = 'if' Expr Block ['else' (IfStatement | Block)].
```

The condition expression is evaluated. It is an error if it does not evaluate to an `int`. If it evaluates to `0`, the `else` block, if any, is executed statement by statement. Otherwise, the `if` block is executed.

There is optional special syntax if the `else` block is itself an `if` statement.

### Loops

A loop executes a body repeatedly while changing the value of a binding:

```
LoopStatement = 'loop' VariableName 'over' Expr Block.
```

The loop expression is evaluated. It is an error if it does not evaluate to a list (in particular, pools cannot be iterated over). A binding for `VariableName` is created in the current environment. For each value in the list, the loop variable is bound to that value, then all statements in the body are executed in succession.

If there are no values in the list, the loop does not execute. If the loop executes at least once, the loop variable remains bound to the last value after the end of the loop.

### Print and output

Both statements allow the user to output a value.

```
PrintStatement = 'print' Expr ['named' Str].
OutputStatement = 'output' Expr OutputOptions.
OutputOptions = ['named' Str ['labeled' Str {',' Str}]
               | 'labeled' Str {',' Str} ['named' Str]].
```

> [!IMPORTANT]
> AnyDice does not support the `print` statement or the `labeled` clause.

It is an error if an `output` statement occurs inside a function. No such restriction exists for the `print` statement.

In both cases, the expression is evaluated. Any strings present as part of a `named` or a `labeled` clause are then interpolated: any pair of square brackets containing a _variable name_ will be replaced by the named variable's value. Square brackets containing any other data will not be replaced. A syntactically valid variable name that is not defined is an error.

> [!IMPORTANT]
> Eurydice is a little more explicit than AnyDice in replaced variables for lists and pools, using strings like `{1, 3, 4}` instead of `{?}`.

For an `output` statement, the value of the expression is converted to a `pool`, and added to an output list.

The optional `named` clause attaches a name to the distribution for display purposes. If no name is provided, the distribution is assigned the default name `output n`, where `n` is its one-based index in program execution order.

The optional `labeled` clause assigns display names to the fields of a tuple-valued output. It is an error to use it with a non-tuple output or to provide a number of labels different from the tuple's arity. The labels do not change the tuple's type or values. `named` and `labeled` may occur in either order.

```
output [tuple d6, d8] labeled "First die", "Second die"
output [tuple d6, d8] named "Joint roll" labeled "d6 result", "d8 result"
```

Enum outputs display member names rather than numeric values. Tuple outputs display every field of each outcome. Numeric statistics and cumulative/order-based presentation do not apply to enum or tuple distributions.

For a `print` statement, the value of the expression is shown to the user as soon as possible, attached to the name if present. There is no default name otherwise.

### Set

The `set` statement allows changing [global settings](#global-settings). It is an error for a `set` statement to occur inside a function.

```
SetStatement = 'set' '"position order"' 'to' ('"highest first"' | '"lowest first"')
             | 'set' '"explode depth"' 'to' Int
             | 'set' '"maximum function depth"' 'to' Int.
```

> [!IMPORTANT]
> Eurydice does not allow arbitrary expressions in `set` statements. Arguments must be integer literals.

### Assignment

Assignment statements either create a new binding in the [innermost environment frame](#values-variables-and-bindings), or replace the value of an existing binding, if a variable with the same name already exists in that frame.

```
AssignmentStatement = VariableName ':' Expr.
```

Assigning to the name of an enum type or enum member is an error. Enum names also cannot be reused for function parameters or loop variables.

### ⊕ Enum definition

```
EnumDefinitionStatement = 'enum' ':' VariableName '{' VariableName {',' VariableName} '}'.
```

Enum definitions are executed sequentially and are only valid at the top level.

### Function definition

Function definitions create a new function binding in the [innermost environment frame](#values-variables-and-bindings), or replace the value of an existing binding, if a function with the same identifier already exists in that frame.

Function definitions are executed sequentially, like other statements. A function call can therefore only resolve a function definition that has already been executed. A later definition with the same identifier replaces the earlier binding from that point onward.

```
FunctionDefinitionStatement = 'function' ':' (Word | ParameterGroup)+ Block.
ParameterGroup = Parameter {',' Parameter}.
Parameter = VariableName [':' Shape].
Shape = 'n' | 's' | 'd'.
```

> [!IMPORTANT]
> AnyDice function definitions can only occur at the top level, and do not accept commas between parameters.

The function's identifier is the sequence of words and argument positions in the name. It is valid for a function identifier to contain no words, or to contain no argument positions.

As in [function calls](#function-calls), adjacent parameters may be separated by commas, and the commas are not part of the identifier. `function: add A:n, B:n` and `function: add A:n B:n` define the same function, and either can be called with or without commas.

Each parameter name can optionally be annotated with a _shape_: `n` for an element, `s` for a list, or `d` for a pool (dice). Note that this constrains the shape only: the value's outcomes may be integers, members of any one enum, or tuples, and a definition cannot constrain their outcome type. Specifying a shape causes the usual argument coercion and pool-based evaluation.

### Return from function

The `result` statement allows returning a value from a function. It is an error for `result` to appear outside a function.

```
ResultStatement = 'result' ':' Expr.
```

Control flow immediately returns to the current function's caller.

## Built-in functions

The following functions are available in the top-level environment frame at the beginning of a program:

### `[absolute N:n]`

Returns the absolute value of `N`.

### ⊕ `[choose FIRST:d if CONDITION:n else SECOND:d]`

Returns `FIRST` when `CONDITION` is nonzero, and `SECOND` when it is zero.

As with other `n`-shaped function parameters, a pool passed as `CONDITION` causes the function to be evaluated for every possible value of the summed pool. The resulting distributions are combined according to the probabilities of those values.

For example, this selects `d6` three quarters of the time and `2d6` one quarter of the time:

```
output [choose d6 if d{0, 1:3} else 2d6]
```

More generally, distributions with nonnegative integer weights `FIRST_WEIGHT` and `SECOND_WEIGHT` can be mixed with:

```
output [choose FIRST if d{0:SECOND_WEIGHT, 1:FIRST_WEIGHT} else SECOND]
```

> [!IMPORTANT]
> The `[choose FIRST:d if CONDITION:n else SECOND:d]` function is not available in AnyDice. For compatible code, it can be defined as:
>
> ```
> function: choose FIRST:d if CONDITION:n else SECOND:d {
>   if CONDITION { result: FIRST }
>   result: SECOND
> }
> ```

### `[SEQ:s contains N:n]`

Returns `1` if `SEQ` contains `N`, or `0` otherwise. `SEQ` and `N` must have the same outcome type.

### `[count NEEDLES:s in HAYSTACK:s]`

Counts the number of occurrences of each element of `NEEDLES` in `HAYSTACK`. Returns the sum of these counts. Both lists must have the same outcome type.

Examples:

```
output [count {1, 2} in {3, 4}]  \ Outputs 0 \
output [count {1, 1, 2} in {1, 2, 2, 3}]  \ Outputs 4 \
```

### ⊕ `[count NEEDLES:s in pool HAYSTACK:d]`

Counts, for every outcome of `HAYSTACK`, the dice whose values occur in
`NEEDLES`. Repeated values in `NEEDLES` count repeatedly. The result is a
dimension-one integer pool, and both arguments must have the same outcome type.

This form operates directly on the pool using the Icepool algorithm, without
first expanding each possible pool outcome into a list.

Examples:

```
output [count {2, 4, 6} in pool 2d6]  \ Outputs d{0:9, 1:18, 2:9} \
output [count {4..6, 6} in pool d6]  \ Outputs d{0:3, 1:2, 2} \
```

### `[explode POOL:d]`

This transforms a die to match an [explosion rule](https://nethackwiki.com/wiki/Exploding_die): if the die rolls its highest face value, that value is kept and the die is rerolled.

The maximum number of rerolls is controlled by the `"explode depth"` [global setting](#global-settings).

If the argument is a pool with dimension >= 1, it is summed first. The return value's dimension is always 1, or 0 if the input pool has dimension 0.

```
output [explode d{}]  \ Outputs d{} \

\ Outputs d{1:9, 2:9, 4:3, 5:3, 7, 8, 9} \
output [explode d{1, 2, 3}]
```

Here is a worked example for `[explode 2d2]` with `explode depth` set to 1:

1. The pool is summed to `d{2, 3:2, 4}`
2. The outcome `4` is mapped to `d{2+4, (3+4):2, 4+4}`, which is the same as the summed die with each value incremented by its largest value. This is equivalent to keeping the 4 and adding the original die again. Other outcomes are left as-is.
3. The final probabilities for each value are:
   * 2: 1/4
   * 3: 1/2
   * 4 and 5: 0
   * 6: 1/4 (probability of rolling an initial 4) * 1/4 (probability of rolling a 2) = 1/16
   * 7: 1/4 (initial) * 1/2 (reroll) = 1/8
   * 8: 1/16 (same as 6)
4. This is equivalent to `d{2: 4, 3: 8, 6, 7: 2, 8}`

### `[explode POOL:d on COND:s]`

This is similar to `[explode POOL:d]`, but instead of exploding on the highest face value, it explodes on any value contained in the `COND` list.

The maximum number of rerolls is controlled by the `"explode depth"` [global setting](#global-settings).

```
output [explode d6 on {1, 6}]  \ Explodes on both 1s and 6s \
output [explode d{1, 2, 3} on {2, 3}]  \ Explodes on 2s and 3s \
```

### ⊕ `[reroll POOL:d]`

This transforms a die to match a reroll rule: if the die rolls its highest face value, the die is rerolled and only the new value is kept (unlike explode, which keeps both the original and new values).

The maximum number of rerolls is controlled by the `"explode depth"` [global setting](#global-settings).

```
output [reroll d6]  \ Rerolls on 6s, keeping only the reroll result \
```

### ⊕ `[reroll POOL:d on COND:s]`

This is similar to `[reroll POOL:d]`, but instead of rerolling on the highest face value, it rerolls on any value contained in the `COND` list.

The maximum number of rerolls is controlled by the `"explode depth"` [global setting](#global-settings).

```
output [reroll d6 on {1, 6}]  \ Rerolls on both 1s and 6s \
output [reroll d{1, 2, 3, 4} on {1, 4}]  \ Rerolls on 1s and 4s \
```

### `[highest COUNT:n of POOL:d]`, `[lowest COUNT:n of POOL:d]`, `[middle COUNT:n of POOL:d]`

These functions return pools of dimension 1 whose distribution is the sum of the highest `COUNT`, lowest `COUNT`, or middle `COUNT` values in the possible outcomes of `POOL`.

`[middle COUNT:n of POOL:d]` uses zero-based indices from `floor((length - COUNT) / 2)`, inclusive, through `floor((length - COUNT) / 2) + COUNT`, exclusive. For example, `[middle 2 of 5d2]` returns a pool corresponding to the second- and third-lowest dice, excluding the first, fourth, and fifth.

### `[highest of FIRST:n and SECOND:n]`, `[lowest of FIRST:n and SECOND:n]`

These functions return the max and min of their two arguments, respectively.

### `[maximum of POOL:d]`

Sums `POOL`, then returns its largest possible outcome as an int.

### `[reverse SEQUENCE:s]`

Returns `SEQUENCE` in reverse order.

### `[sort SEQUENCE:s]`

This function returns a list with the same elements as `SEQUENCE`, sorted in ascending or descending order depending on the value of the `"position order"` global setting.

The list must have `int` outcomes; enums and tuples have no language-level ordering.

### ⊕ `[tuple A:n B:n]`, `[tuple A:n B:n C:n]`, `[tuple A:n B:n C:n D:n]`

Constructs a tuple from two, three, or four `Scalar` fields; nested tuples are an error.

Pool-valued arguments are expanded using normal [pool-based evaluation](#pool-based-evaluation), producing a joint tuple distribution.

### ⊕ `[field INDEX:n of TUPLE:n]`

Returns the field at the one-based `INDEX` in `TUPLE`. `INDEX` must be an `int`, `TUPLE` must be a tuple, and an out-of-bounds index is an error.

If `TUPLE` is a pool, normal [pool-based evaluation](#pool-based-evaluation) projects the field from every tuple outcome while preserving its probabilities.

## Acknowledgements

The Eurydice language is largely compatible with [AnyDice by Jasper Flick](https://anydice.com/). Thanks!

This spec is inspired by the [Starlark spec](https://github.com/bazelbuild/starlark/blob/master/spec.md).
