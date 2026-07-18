# Eurydice language specification

This specification aims to provide a description of how Eurydice should behave. There may be divergences between Eurydice and this description - those are bugs, please report them!

This document is intended mostly as a specification for other AnyDice implementers, or for power users. If you're looking for an introduction to the AnyDice DSL, this document is likely too technical. Consider looking at [AnyDice docs](https://anydice.com/docs/) for a smoother introduction.

**This project is not affiliated with AnyDice or Jasper Flick.**

## Overview

Eurydice is a dialect of the [AnyDice](https://anydice.com/) language. It is almost completely compatible with AnyDice, with some extensions, and a few restrictions.

> [!IMPORTANT]
> Intentional divergences from AnyDice are marked with information boxes like this one.

There may also be unintentional divergences between this document and AnyDice's, or Eurydice's, behavior. Please report those on GitHub.

Eurydice is reproducible and hermetic: programs can't interact with the environment, and two runs will have exactly the same results.

This report frequently uses the following notations to describe pools:

* `<n>d<k>`: represents a pool of `n` die with `k` faces labeled 1..k
* `<n>d{<k1>:<count1>,...,<kn>:<countn>}`: represents a pool of `n` die with `count1` faces showing `k1`, `count2` faces showing `k2`, etc., and `countn` faces showing `kn`. If `:<count>` is omitted for any `k`, the count is assumed to be 1.

This is exactly compatible with Eurydice expressions using the `d` operator.

## Lexical elements

A Eurydice program is composed of a sequence of statements. Eurydice is whitespace-insensitive, except to separate tokens.

_Block comments_ start and end with backslashes: `\ this is a comment \`.

_Line comments_ start with a triple backslashes: `\\\ this is a line comment`.

> [!Important]
> AnyDice does not support line comments.

Comments are treated as whitespace, and can be used to separate tokens, e.g. `[hello\a comment\world]` is equivalent to `[hello world]`.

_Variable identifiers_ are composed of uppercase letters and underscores, for instance `X`, `MY_VARIABLE`, or `_MY_VARIABLE`. `MyVariable`, or `x` are not variable identifiers.

_Words_ are composed of lowercase letters and underscores (after the first position only), except the single lowercase letter `d`, which is punctuation and not a word. `word`, `if`, `output`, `my_word` are all words. `WORD`, `wOrd`, and `_word` are not.

> [!IMPORTANT]
> AnyDice words (function names) can't contain underscores.

_Punctuation_: the following characters or sequences of characters are tokens:

```
+   =   &   @   :   (   ..
-   !=  |   d   [   )
*   <   !   #   ]   {
/   <=              }
^   >
    >=
```

_Integer literals_: Integer literals are base-10 representations of an unsigned 31-bit integer. It is an error if an integer literal represents a number greater than 2^31-1 (2147483647). This is a subset of the values of the underlying `int` type, which can represent 32-bit signed integers (negative values can be obtained by using the unary minus operator). Examples include `0`, `123`.

_String literals_: String literal can only occur in a few places. They are enclosed by double quotes (`"`). A string literal can contain escaped double quotes: `"message: \"hello\""`. No other escape sequences are supported.

## Program execution

A Eurydice program is composed of a sequence of statements. Statements are executed sequentially as specified. Once the program has finished executing, all outputs are displayed to the users.

Errors are fatal and terminate program execution.

## Data types

There are two types of scalars in Eurydice:

* `int`: a 32-bit signed integer
* _enums_ are user-declared types which each contain a set of nominal, non-numeric values, which can be used to represent discrete outcomes, booleans, etc.

> [!IMPORTANT]
> AnyDice does not support enums; all scalar values are numeric.

And three types of values built on these basic types:

* scalars: hold a single `int` or enum value
* `list`: values of this type hold a list of scalars.
* `pool`: values of this type hold a pool, which is composed of a mapping of outcomes (each of which is a scalar) to probabilities (whose representation is unspecified), together with an unsigned count of dice, which is called the _dimension_.

Scalars, lists, and pools have an _outcome type_: either `int` or one particular enum. Lists and pools are homogeneous and cannot mix integers with enum members or members of different enums. An empty literal is numeric unless its enum type comes from an enum-typed expression, such as `{MISS:0}`; an untyped empty literal is not inferred to satisfy an enum constraint.

The maximum number of elements in a list, or of outcomes in a pool, is 2^31-1.

> [!IMPORTANT]
> AnyDice does not restrict integers, or pool outcomes, to 32-bit values. Experimentation suggests that AnyDice outcomes are represented as double-precision floats (`output 9007199254740993` returns 9007199254740992).

### Enums

Enums are declared at the top level:

```
enum: ATTACK_RESULT { MISS, HIT, CRITICALHIT }
```

The enum type and its members use variable-identifier syntax. An enum must contain at least one member. Enum type and member names share a namespace and cannot collide. Both are immutable; members are global constants. Enum declarations cannot occur in functions or nested blocks and cannot replace an existing enum, member, or variable binding.

Enum members support equality and inequality with members of the same enum. Arithmetic, ordering, boolean negation, ranges, numeric dice-side or dice-count use, sorting, and other numeric operations are errors. Homogeneous enum values can be placed in lists and dimension-one pools, passed to functions, returned, repeated, reversed, counted, tested for containment, and displayed. Enum pools are categorical distributions, not collections of multiple sampled enum values: `d{MISS, HIT}` is valid, while `2d{MISS, HIT}` is an error. Use an enum sequence to represent multiple enum values.

There are no first-class functions.

### Pools

It is not possible to create a pool value representing a pool of different types of dice. For instance, `3d6` is a pool of three d6s, but there is no way to represent a pool of one d6 and one d8.

Pools may have no possible outcomes (for instance, such a pool is created with expression `d{}`). Such a pool has dimension 0.

The following operations are frequently referred to in this document:

#### Summing

_Summing_ a pool transforms it into a pool of dimension 1 whose outcomes are the possible sums when sampling from the dice in the pool, with associated probabilities. Summing a pool of dimension 1 does nothing. Summing a pool of dimension 0 creates a pool of dimension 1 with a single outcome, 0.

For instance, summing `2d2` results in a pool equivalent to `d{2, 3, 3, 4}`.

#### Multiset iteration

_Iterating over the multisets_ in a pool refers to processing each possible _sorted_ outcome of the collection of dice in the pool, together with its associated probability. For instance, the multisets of `3d{1, 2:2, 3:3}` are:

```
multiset              probability
{1, 1, 1}:   (1/6)^3                     * 6
{1, 1, 2}:   (1/6)^2 *  1/3              * 2
{1, 1, 3}:   (1/6)^2           *  1/2    * 2
{1, 2, 2}:    1/6    * (1/3)^2           * 2
{1, 2, 3}:    1/6    *  1/3    *  1/2    * 1
{2, 2, 2}:             (1/3)^3           * 6
{2, 2, 3}:             (1/3)^2 *  1/2    * 2
{2, 3, 3}:              1/3    * (1/2)^2 * 2
{3, 3, 3}:                       (1/2)^3 * 6
```

The probability associated with each multiset element is equal to the product of each element's probability, multiplied by the multiplicity factor for the outcome (which is $\frac{(\mathrm{size\ of\ set})!}{\mathrm{multinomial\ coefficient}}$).

If there is a single die in the pool, this is the same as iterating over the die's outcomes.

#### Outcome mapping

_Outcome mapping_ a pool transforms each multiset of a pool with a specified operation. If two outcomes are transformed into the same value, their probabilities are summed.

For instance, outcome mapping `2d{-1..1}` with the operation `x→2*abs(x)` results in a pool equivalent to `d{0:3, 2:4, 4:2}`.

#### Flat mapping

_Flat mapping_ a pool uses a function that turns an outcome into a pool. The function is called for each multiset of the original pool. The resulting pools are combined together, respecting the probability of the original outcome that produced them.

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

Starting with pools $p_1, p_2, \ldots, p_n$, the _multiset cross-product_ is an iterator over pairs of (list of multiset, probability), where:

* The lists of multisets range over all possible combinations of multisets from each of the pool's multiset iterators
* The probability associated with each list is the product of the multisets' probabilities.

In pseudocode:

```
for (multiset_1, prob_1) in p_1.multisets():
    ...
        for (multiset_n, prob_n) in p_n.multisets():
            yield ([multiset_1, ..., multiset_n], prob_1 * ... * prob_n)
```

## Values, variables, and bindings

All Eurydice _values_ are immutable: it is not possible to modify a list or a pool in place.

Eurydice uses [dynamic scoping](<https://en.wikipedia.org/wiki/Scope_(computer_science)#Lexical_scope_vs._dynamic_scope>). At every point during execution, an _environment_ maps variable names to values, and function names to functions. Outside a function, the environment is known as the _global environment_. When execution is about to enter a function's body, a new _environment frame_ is pushed to the stack of environments. This environment frame is popped and deleted when execution leaves the function.

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
function x {
    result: X
}

function reassign x {
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

* `explode depth`: set to a non-negative number, default 2.
* `maximum function depth`: set to a non-negative number, default 10. An attempt to call a function when the current recursion depth is already equal to the maximum function depth will result in the function call not actually executing, and returning an empty list.
* `position order`: set to `"lowest first"` or `"highest first"` (default `"highest first"`). This setting affects three things:
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
Y  \\\ Error
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
ListElem = (Expression|Expression '..' Expression) [':' Expression].
```

Expressions composing a list elements are evaluated in an unspecified order. If the list element is a range, both expressions must evaluate to `int` values, and the range is replaced by a list containing all integers between the start and end of the range, inclusive. If the start of the range is greater than the end, an empty list is produced.

If a repeat expression is present, it must evaluate to an `int`. If this value is negative, it is replaced with `0`. The primary expression is then [flattened into a list](#list-flattening), and is concatenated to itself the indicated number of times. If the repeat count is 0, the empty list is produced.

#### List flattening

List flattening transforms values in the following way:

* `int` values are flattened by enclosing them in a single-element list.
* `list` values are unchanged by flattening.
* `pool` values are flattened first by [summing](#summing), then by discarding the probabilities and creating a list containing each outcome in ascending order. Outcomes with nonzero probability appear once regardless of their probabilities.

#### Examples

```
output {}                \\ Outputs {}
output {1, 2, 3}         \\ Outputs {1, 2, 3}
output {d4}              \\ Outputs {1, 2, 3, 4}
output {d2:2}            \\ Outputs {1, 2, 1, 2}
output {{1, 2}, {3, 4}}  \\ Outputs {1, 2, 3, 4}
```

### Parenthesized expressions

```
'(' Expr ')'.
```

A single expression enclosed in parentheses yields the result of that expression. Explicit parentheses may be used for clarity, or to override the default association of subexpressions.

### Unary operators

There are 4 unary operators, which all bind tighter than any binary operator. There are `!`, `-`, `#`, and `d`.

`!` and `-` compute the logical and arithmetic negation of their argument.

* If the argument is an `int`, `-` negates the value, while `!` evaluates 0 if the argument is nonzero, and 1 otherwise.
* If the argument is a `list`, the values of the list are summed to produce an int, then `!` or `-` is applied to that int to produce the result.
* If the argument is a `pool`, the pool is summed, then the [outcomes of the pool are all mapped](#outcome-mapping) with the operator.

`#` evaluates to the length of its argument:

* If the argument is an `int`, it returns the number of digits in the base-10 representation. `#0` evaluates to `1`. A leading `-` is not counted as a digit.
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

1. The RHS operand is converted to a pool with the following operation.
   1. If it is an `int` `i`, it is converted to a pool with values 1 to `abs(i)` inclusive (if `i` is 0, the pool contains the single outcome 0). If `i` is negative, the pool outcomes then are mapped to their opposite. For instance, `d(-3)` evaluates to `d{-1, -2, -3}`.
   2. If it is a `list`, it is converted to a pool whose outcomes are the distinct values in the list, and whose probability for each outcome is proportional to the number of occurrences of each value in the list.
   3. Pools provided as an RHS operand are not transformed.
2. The LHS operand is summed if it is a `list`, resulting in either an `int` or a `pool`.
   1. If it is an `int` `i`, the dimension of the RHS pool is multiplied by `abs(i)`. If `i` is negative, then each outcome in the resulting pool is multiplied by `-1`.
   2. If it is a `pool`, then the RHS is [flat mapped](#flat-mapping) with the operation described in (a). (Recall that flat-mapping takes a pool and an `int -> pool` function; the operation described in (a) is such a function).

#### `@` operator

The `@` operator selects the (LHS)-th element from its RHS.

First, if the LHS argument is an int, it is converted to a singleton list. It is an error if the LHS argument is a pool.

* If the RHS is an `int` or `list`, for each value `i` in the resulting list:
  * If the RHS is an `int` `j`, the base-10 digit of `abs(j)` at index `i` is selected. If `j` is negative, this digit is multiplied by -1. If the `"position order"` [global setting](#global-settings) is set to `"highest first"`, index 1 corresponds to the most significant digit; otherwise, to the least significant. Valid indices start at 1.
  * If the RHS is a `list`, its element at position `i` is selected. The first element of the list has index 1. This is not affected by the `"position order"` setting.
  * In both the int and list RHS cases, if `i` is invalid (zero or negative, or greater than list length or digit count), the expression evaluates to 0.
  * Finally, all selected elements are summed to produce an `int`.
* If the RHS argument is a `pool`, then, each outcome multiset is flat mapped with the following function:
  * The multiset is sorted according to the `"position order"` [global setting](#global-settings).
  * Elements are selected from the multiset and summed as if using the `@` operator from a list.

#### Mathematical operators

The operators `^`, `*`, `/`, `+`, `-`, `&`, and `|` are mathematical operators, performing exponentiation, multiplication, integer division, addition, subtraction, logical conjunction, and logical disjunction, respectively.

Applied to `int`s, `&` evaluates to `1` if both of its arguments are nonzero, `0` otherwise; `|` evaluates to `1` if at least one argument is nonzero. Other mathematical operators have standard behavior on the integers.

Division by 0 causes an error to be raised. `0^0` evaluates to 1.

1. If either argument to a mathematical operator is a `list`, it is summed to an `int`.
2. After this, if both arguments are `int`s, the operator expression evaluates to an `int`.
3. Otherwise, both arguments are converted to `pool`s and summed. The LHS is then flat mapped with a function that applies the operator to the LHS and each value in the RHS.

#### Comparison operators

The operators `=`, `!=`, `<`, `<=`, `>`, and `>=` are comparison operators, performing equality, inequality, smaller-than, smaller-than-or-equal, larger-than, and larger-than-or-equal checks respectively.

In all cases, these operators evaluate to `1` if their condition is true, or `0` otherwise.

1. If both arguments are `int`s, the operator expression evaluates to an `int`.
2. If one argument is a `list`, and the other is an `int`, the comparison is performed between the `int` and each member of the `list`. The expression's value is an `int`: the count of comparisons that evaluated to true.
3. If both arguments are `list`s, they are compared in [lexicographic order](https://en.wikipedia.org/wiki/Lexicographic_order).
4. If either argument is a `pool`, both arguments are converted to `pool`s and summed. The LHS is then flat mapped with a function that applies the operator to the LHS and each value in the RHS.

### Function calls

Functions can be called by enclosing the words that make up their names, interleaved with expressions, between square brackets:

```
FunctionCallItem = Word | Expr SafeExpr {SafeExpr} Word.
FinalFunctionCallItem = Expr SafeExpr {SafeExpr}.
FunctionCall = '[' {FunctionCallItem} [FinalFunctionCallItem] ']'.
```

A technical trick is needed in the grammar because arguments are allowed to follow each other with no word in between: `SafeExpr` is any expression that does not start with the tokens `-` or `d`. Because expressions can immediately follow each other, it would otherwise not be possible to know if a call like `[f 1 -2]` should be parsed as `[f (1) (-2)]` or as `[f (1-2)]`. An `Expr` in the function call must be followed by either a `Word`, which clearly separates the two arguments, or a `SafeExpr`, which cannot be a suffix of an `Expr`, and is thus necessarily a new argument.

To evaluate a function call, the function identifier is resolved first. A function is identified by the sequence of words and argument locations in its name, and it is an error if the function's identifier is not in scope in the current environment. Here are some examples:

```
function: add A:n B:n { A + B }
output [add 1 2]  \\\ outputs 3
output [add 1 - 2]  \\\ error

function: add A:n and B:n { A + B }
output [add 1 and 2]  \\\ outputs 3
output [add 1 and -2]   \\\ outputs -1
```

From then, function calls proceed in several steps.

#### Argument coercion

The actual types of the argument expressions are compared to the expected argument types, and each is transformed in the following way:

* If the expected argument type is unspecified, or if the expected type matches the actual type, the value is not transformed.
* If the actual argument is a `list`:
  * If an `int` is requested, the `list` is summed to an `int`.
  * If a `pool` is requested, a `pool` is created, with equally likely outcomes from the `list`.
* If the actual argument type is an `int`:
  * If a `list` is requested, a singleton list is created.
  * if a `pool` is requested, a single-outcome `pool` is created.
* If the actual argument type is a `pool`, and an `int` is requested, the pool is summed, creating a new `pool` (see note below: `pool`-typed values can be passed to `int`-typed arguments).

After this process, some values of type `pool` may still correspond to arguments where `int` or `list` types are requested. If this is not the case, the function is called once, and the value of the expression is the result of [evaluating the function](#function-evaluation). If it is the case, evaluation proceeds as described in the next section.

#### Pool-based evaluation

This section applies if any `pool` values are being passed to `int` or `list` typed arguments. As mentioned above, any `pool`s corresponding to an `int` argument is summed to dimension 1. Then, the [multiset cross product](#multiset-cross-product) of the pools is generated. The order in which elements inside each multiset are ordered follows the `"position order"` [global setting](#global-settings).

The function is then evaluated once for each value in the multiset cross product (this may be 0 times if the cross product is empty). In each invocation, the argument values are:

* For any argument whose actual type correspond to its declared type, the actual value.
* For arguments where a `pool` was provided but a `list` or `int` was requested, that pool's multiset value in the current element of the multiset cross product iterator.

Each invocation's result is converted to a `pool` through the standard conversions. The final result of the function call is a `pool` created by iterating through every possible outcome of the intermediate pools, and summing the probability of that outcome in each of the intermediate pools, multiplied by the probability of the value that generated this intermediate pool.

For instance, suppose the following call needs to be evaluated:

```
function: one or N:n th die from FACES:s {
  result: d{1, N@FACES}
}
output [one or d3 th die from 3d2]
```

The cross-product multisets and their corresponding intermediate pools are:

| Multiset for `N` | Multiset for `FACES` | Weight | Outcome | Weight for `1` | Weight for `2` |
|------------------|----------------------|--------|---------|----------------|----------------|
| [1]              | [1, 1, 1]            | 1      | d{1}    | 1              |                |
| [2]              | [1, 1, 1]            | 1      | d{1}    | 1              |                |
| [3]              | [1, 1, 1]            | 1      | d{1}    | 1              |                |
| [1]              | [1, 1, 2]            | 3      | d{1}    | 3              |                |
| [2]              | [1, 1, 2]            | 3      | d{1}    | 3              |                |
| [3]              | [1, 1, 2]            | 3      | d{1, 2} | 3/2            | 3/2            |
| [1]              | [1, 2, 2]            | 3      | d{1}    | 3              |                |
| [2]              | [1, 2, 2]            | 3      | d{1, 2} | 3/2            | 3/2            |
| [3]              | [1, 2, 2]            | 3      | d{1, 2} | 3/2            | 3/2            |
| [1]              | [2, 2, 2]            | 1      | d{1, 2} | 1/2            | 1/2            |
| [2]              | [2, 2, 2]            | 1      | d{1, 2} | 1/2            | 1/2            |
| [3]              | [2, 2, 2]            | 1      | d{1, 2} | 1/2            | 1/2            |

The final result has outcome 1 with weight 18, and outcome 2 with weight 6.

#### Function evaluation

A [new environment frame](#values-variables-and-bindings) is created, binding the value of each expression to its corresponding argument name. Each statement in the function's body is then evaluated sequentially. If a [return statement](#return-from-function) is encountered, execution returns to the caller, and the function call's value is the return statement's expression.

If no return statement is encountered, the function returns an empty list.

## Statements

```
Statement = IfStatement | LoopStatement | PrintStatement | OutputStatement
          | FunctionDefinitionStatement | ResultStatement | AssignmentStatement
          | EnumDefinitionStatement | SetStatement.
Block = '{' Statement* '}'.
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
LoopStatement = `loop` VariableName `over` Expr Block.
```

The loop expression is evaluated. It is an error if it does not evaluate to a list (in particular, pools cannot be iterated over). A binding for `VariableName` is created in the current environment. For each value in the list, the loop variable is bound to that value, then all statements in the body are executed in succession.

If there are no values in the list, the loop does not execute. If the loop executes at least one, the loop variable remains bound to the last value after the end of the loop.

### Print and output

Both statements allow the user to output a value.

```
PrintStatement = 'print' Expr ['named' Str].
OutputStatement = 'output' Expr ['named' Str].
```

> [!IMPORTANT]
> AnyDice does not support the `print` statement.

It is an error if an `output` statement occurs inside a function. No such restriction exists for the `print` statement.

In both cases, the expression is evaluated. If present, the name string is then interpolated: any pair of square brackets containing a _variable name_ will be replaced by the named variable's value. Square brackets containing any other data will not be replaced. To be replaced, a variable name must correspond to an actually defined variable; otherwise, the bracketed string is left as-is.

> [!IMPORTANT]
> Eurydice is a little more explicit than AnyDice in replaced variables for lists and pools, using strings like `{1, 3, 4}` instead of `{?}`.

For an `output` statement, the value of the expression is converted to a `pool`, and added to an output list. If a name is not provided, it is associated with the default name `output n`, where `n` is the 1-indexed output number.

Enum outputs display member names rather than numeric values. Numeric statistics and cumulative/order-based presentation do not apply to enum distributions.

For a `print` statement, the value of the expression is shown to the user as soon as possible, attached to the name if present. There is no default name otherwise.

### Set

The `set` statement allows changing [global settings](#global-settings). It is an error for a `set` statement to occur inside a function.

```
SetStatement = 'set' '"position order"' 'to' '"highest first"'.
SetStatement = 'set' '"position order"' 'to' '"lowest first"'.
SetStatement = 'set' '"explode depth"' 'to' Int.
SetStatement = 'set' '"maximum function depth"' 'to' Int.
```

> [!IMPORTANT]
> Eurydice does not allow arbitrary expressions in set statements. Arguments must be int literals.

### Assignment

Assignment statements either create a new binding in the [innermost environment frame](#values-variables-and-bindings), or replace the value of an existing binding, if a variable with the same name already exists in that frame.

```
AssignmentStatement = VariableName ':' Expr.
```

Assigning to the name of an enum type or enum member is an error. Enum names also cannot be reused for function parameters or loop variables.

### Enum definition

```
EnumDefinitionStatement = 'enum' ':' VariableName '{' VariableName {',' VariableName} '}'.
```

Enum definitions are executed sequentially and are only valid at the top level. Consequently, an enum type used by a function annotation must be declared before that function definition is executed.

### Function definition

Function definitions create a new function binding in the [innermost environment frame](#values-variables-and-bindings), or replace the value of an existing binding, if a function with the same identifier already exists in that frame.

Function definitions are executed sequentially, like other statements. A function call can therefore only resolve a function definition that has already been executed. A later definition with the same identifier replaces the earlier binding from that point onward.

```
FunctionDefinitionStatement = 'function' ':' (Word | Parameter)+ Block.
Parameter = VariableName [':' Type].
Type = 'n' | 's' | 'd' | OutcomeType | 's' '<' OutcomeType '>' | 'd' '<' OutcomeType '>'.
OutcomeType = 'int' | VariableName.
```

> [!IMPORTANT]
> AnyDice function definitions can only occur at the top level.

The function's identifier is the sequence of words and argument positions in the name. It is valid for a function identifier to contain no words, or to contain no argument positions.

Each argument name can optionally be annotated with a type. `n`, `s`, and `d` constrain only the value's shape to scalar, sequence, and pool respectively; their outcomes may be integers or members of any one enum. `int`, `s<int>`, and `d<int>` additionally require numeric outcomes. An enum name, `s<ENUM>`, or `d<ENUM>` requires outcomes of that particular enum. Specifying a shape causes the usual argument coercion and pool-based evaluation, while incompatible outcome types produce an error.

An enum scalar may be coerced to a singleton sequence or pool. An enum pool may be expanded for an enum scalar or sequence parameter. Enum sequences cannot be summed into enum scalars, and enum pools always have dimension one.

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

### `[SEQ:s contains N:n]`

Returns `1` if `SEQ` contains `N`, or `0` otherwise.

### `[count NEEDLES:s in HAYSTACK:s]`

Counts the number of occurrences of each element of `NEEDLES` in `HAYSTACK`. Returns the sum of these counts.

Examples:

```
output [count {1, 2} in {3, 4}]  \ Outputs 0 \
output [count {1, 1, 2} in {1, 2, 2, 3}]  \ Outputs 4 \
```

### `[explode POOL:d]`

This transforms a die to match an [explosion rule](https://nethackwiki.com/wiki/Exploding_die): if the die rolls its highest face value, that value is kept and the die is re-rolled.

The maximum number of rerolls is controlled by the `"explode depth"` [global setting](#global-settings).

If the argument is a pool with dimension >= 1, it is summed first. The return value's dimension is always 1, or 0 if the input pool has dimension 0.

```
output [explode d{}]  \ Outputs d{} \

\ Outputs d{1:9, 2:9, 4:3, 5:3, 7, 8, 9} \
output [explode d{1, 2, 3}]
```

Here's a worked out example for `[explode 2d2]` with `explode depth` set to 1:

1. The pool is summed to `d{2, 3:2, 4}`
2. The outcome `4` is mapped to `d{2+4, (3+4):2, 4+4}`, which is the same as the summed die with each value incremented by its largest value. This is equivalent to keeping the 4 and adding the original die again. Other outcomes are left as-is.
3. The final probabilities for each value are:
   * 2: 1/4
   * 3: 1/2
   * 4 and 5: 0
   * 6: 1/4 (probability of rolling an initial 4) * 1/4 (probability of rolling a 2) = 1/16
   * 7: 1/4 (initial) * 1/2 (reroll) = 1/8
   * 8: 1/16 (same as 6)
4. This is equivalent to `d{2: 4, 3: 8, 6, 7: 2, 8: 4}`

### `[explode POOL:d on COND:s]`

This is similar to `[explode POOL:d]`, but instead of exploding on the highest face value, it explodes on any value contained in the `COND` list.

The maximum number of rerolls is controlled by the `"explode depth"` [global setting](#global-settings).

```
output [explode d6 on {1, 6}]  \ Explodes on both 1s and 6s \
output [explode d{1, 2, 3} on {2, 3}]  \ Explodes on 2s and 3s \
```

### `[reroll POOL:d]`

This transforms a die to match a reroll rule: if the die rolls its highest face value, the die is re-rolled and only the new value is kept (unlike explode, which keeps both the original and new values).

The maximum number of rerolls is controlled by the `"explode depth"` [global setting](#global-settings).

```
output [reroll d6]  \ Rerolls on 6s, keeping only the reroll result \
```

> [!IMPORTANT]
> The `[reroll POOL:d]` function is not available in AnyDice.

### `[reroll POOL:d on COND:s]`

This is similar to `[reroll POOL:d]`, but instead of rerolling on the highest face value, it rerolls on any value contained in the `COND` list.

The maximum number of rerolls is controlled by the `"explode depth"` [global setting](#global-settings).

```
output [reroll d6 on {1, 6}]  \ Rerolls on both 1s and 6s \
output [reroll d{1, 2, 3, 4} on {1, 4}]  \ Rerolls on 1s and 4s \
```

> [!IMPORTANT]
> The `[reroll POOL:d on COND:n]` function is not available in AnyDice.

### `[highest COUNT:n of POOL:d]`, `[lowest COUNT:n of POOL:d]`, `[middle COUNT:n of POOL:d]`

These functions return pools of dimension 1 whose distribution is the sum of the highest `COUNT`, lowest `COUNT`, or middle `COUNT` values in the possible outcomes of `POOL`.

`[middle COUNT:n of POOL:d]` returns indices between ((length-COUNT) / 2) inclusive and ((length-COUNT) / 2 + COUNT) exclusive, rounding the division down, and indexing from 0. For instance, `[middle 2 of 5d2]` will return a pool corresponding to the 2nd and 3rd lowest dice, excluding the 1st, 4th and 5th.

### `[highest of FIRST:n and SECOND:n]`, `[lowest of FIRST:n and SECOND:n]`

These functions return the max and min of their two arguments, respectively.

### `[maximum of POOL:d]`

Sums `POOL`, then returns its largest possible outcome as an int.

### `[reverse SEQUENCE:s]`

Returns `SEQUENCE` in reverse order.

### `[sort SEQUENCE:s]`

This function returns a sequence with the same elements as `SEQUENCE`, sorted in ascending or descending order depending on the value of the `"position order"` global setting.

## Acknowledgements

The Eurydice language is almost entirely identical to [AnyDice by Jasper Flick](https://anydice.com/). Thanks!

This spec is inspired by the [Starlark spec](https://github.com/bazelbuild/starlark/blob/master/spec.md).
