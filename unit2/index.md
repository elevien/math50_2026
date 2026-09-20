---
layout: notes
title: "Unit 2: Expectation, the Normal Distribution & Regression"
unit_title: "Unit 2"
unit_subtitle: "Expectation, the Normal distribution, and the conditionally Normal regression model"
toc:
  - {href: "#sec-2-1", label: "2.1 Expectation & variance"}
  - {href: "#sec-2-2", label: "2.2 The Normal distribution"}
  - {href: "#sec-2-3", label: "2.3 Linear regression: a conditionally Normal model"}
  - {href: "#problems", label: "Problems"}
---

<div class="unit-overview" markdown="1">

In this unit we introduce **expectation**, an operation which takes a random variable and produces a deterministic quantity. The expectation of a random variable can be approximated with **sample averages**, and from them we can infer properties of the model (like its parameters). Much of statistics relies on the basic fact that sample averages approximate expectations, something we will learn more about (when discussing estimators) in Unit 3. Later in the unit we discuss the **Normal distribution** and a probability model built from it: a **linear regression** model. 

#### Concepts

Expectation, variance and standard deviation, conditional expectation, empirical averages, coefficient of variation, linearity of expectation, the tower property and the law of total variance, the Normal distribution, standardization, linear transformations and sums of independent Normal random variables, the single-predictor linear regression model, exogenous vs. endogenous predictors.

#### Things to practice

- Calculate expectations by hand and translate between mathematical definitions and code (on exams you will not be asked to write code, but you may be asked to explain what a few lines of code do).
- Work with the Normal distribution: standardize a variable, compute probabilities using the bell curve shape (68/95/99.7), and find the distribution of a linear transformation or sum of independent Normal random variables.
- For a linear regression model with a binary predictor, understand why the regression coefficient is a difference of conditional averages.
- State the assumptions of a linear regression model and identify (e.g. from plots) when they are violated.
- Recognize when a predictor may be endogenous, meaning the model's error term may still be related to the predictor.

</div>

<p class="pdf-link"><a href="unit2.pdf">Unit 2 notes pdf</a> <a href="https://colab.research.google.com/github/elevien/math50_2026/blob/main/unit2/unit2.ipynb">Unit 2 notebook</a></p>

## 2.1 Expectation, variance and standard deviation {#sec-2-1}

### Sample averages and expectation

It's usually difficult to obtain the full distribution of a random variable from data, and it may not even be that relevant to the questions we're asking. Instead, we'd like to summarize properties of a random variable by looking at averages. In other cases, we have a good idea what type of distribution we're dealing with, but there are unknown parameters that can be estimated by averages.

You're probably already familiar with the <span class="term">[sample mean](https://en.wikipedia.org/wiki/Sample_mean_and_covariance)</span> (also called the <span class="term">sample average</span> or <span class="term">empirical average</span>). If $Y_1,Y_2,\dots,Y_n$ are iid samples of $Y$, the sample mean is

$$ \overline{Y} = \frac{1}{n}\sum_{i=1}^n Y_i. $$

In code, if `x` is a NumPy array holding $Y_1,\dots,Y_n$, the sample mean is `np.mean(x)`.

More generally, we might look at the average of some function of a random variable,

$$ \overline{g(Y)} = \frac{1}{n}\sum_{i=1}^n g(Y_i) $$

(a function of a random variable is just another random variable, so there's nothing too deep here). If we take

$$ g(y) = 1_A(y) = \begin{cases} 1 & y \in A \\ 0 & y \notin A \end{cases} $$

then we connect the idea of a sample average to an estimate of a probability:

$$ \overline{g(Y)} = \frac{N(y \in A)}{n} \approx P(\lbrace Y \in A \rbrace). $$

Any quantity we compute from data is, in some sense, a sample average &mdash; so a great deal of statistics is about understanding the behavior of sample averages.

Now we introduce the idea of <span class="term">[expectation](https://en.wikipedia.org/wiki/Expected_value)</span>. It's helpful to think of expectation as the mathematical idealization of a sample average, just as probabilities are mathematical idealizations of long-run frequencies.

<details markdown="1">
<summary>Frequentist vs. Bayesian interpretation</summary>

Recall the flipped-but-hidden coin from [Unit 1](../unit1/#sec-1-3). Whether you interpret its probability of heads as a long-run frequency or as your degree of belief (given the outcome is already fixed) is a key distinction in statistics. Thinking of probability as a long-run frequency is the frequentist interpretation and belief is Bayesian.  We will say more about this in Units 6 and 7.
</details>

Suppose each $Y_i$ is iid with sample space $S$. If $n$ is large, the fraction of samples for which $Y_i=y$ will be $\approx P(\lbrace Y=y \rbrace)$. We can express the sample average in terms of probabilities:

$$ \overline{Y} = \frac{1}{n}\sum_i Y_i = \frac{1}{n}\sum_{y\in S} y\,N(Y=y) $$

$$ = \sum_{y \in S} y\,\frac{N(Y=y)}{n} \approx \sum_{y \in S} y\,P(Y=y). $$

The right-hand side is the definition of the mean, or expectation, denoted

$$ E[Y] = \sum_{y} y\,P(Y=y). $$

To summarize what we just saw (which should be intuitively clear):

$$ E[Y] \approx \overline{Y}. $$

Sometimes we write $\mathbb E$ instead of $E$ to distinguish it from other variables named $E$. If we have a function $g: S \to S'$ from the sample space to some other space, then $g(Y)$ is a new random variable with sample space $S'$ &mdash; but we don't usually need the distribution of $X=g(Y)$ to compute its expectation, since

$$ E[X] = E[g(Y)] = \sum_{y \in S} g(y)\,P(Y=y). $$

This is convenient: we can use the distribution of $Y$ to compute the expectation of $X$, which is often simpler (this will matter for the variance, below).

It's important to understand that, just like probabilities, expectation is an operation that takes a random variable to a deterministic number. The sample average is the approximate, empirical version of this &mdash; I like to think of expectations and sample averages as living in "math world" and "data world" respectively. In Unit 3 we'll make the connection between the two worlds precise via the law of large numbers and the CLT.

<div class="example" markdown="1">
#### Example (expectation of a discrete random variable)

Let $Y$ take values in $\lbrace 1,2,3 \rbrace$ with $P(Y=1)=\tfrac12$, $P(Y=2)=\tfrac13$, $P(Y=3)=\tfrac16$.

<u>Question:</u> what is $E[Y]$?

<u>Solution:</u>

$$ E[Y] = \sum_{y=1}^3 y\,P(Y=y) = 1\cdot\tfrac12 + 2\cdot\tfrac13 + 3\cdot\tfrac16 = \tfrac12+\tfrac23+\tfrac12 = \tfrac53. $$

So $E[Y] = 5/3$.

The code below illustrates the sample mean converging to $5/3$ as $n$ grows.

```python
import numpy as np
import matplotlib.pyplot as plt

values = [1, 2, 3]
probs = [1/2, 1/3, 1/6]

Nmax = 5000
samples = np.random.choice(values, size=Nmax, p=probs)

sample_means = np.cumsum(samples) / np.arange(1, Nmax + 1)

plt.figure(figsize=(6,4))
plt.plot(sample_means, label="Sample mean")
plt.axhline(5/3, color="red", linestyle="--", label="Theoretical $E[Y]=5/3$")
plt.xlabel("n (sample size)")
plt.ylabel("Sample mean")
plt.legend()
plt.title("Convergence of sample mean to expectation")
plt.show()
```
</div>

This same convergence appears in the [Unit 1 Bernoulli simulation](../unit1/#sec-1-3): the running sample frequency is just the sample mean of Bernoulli draws, so it settles near $E[Y]=q$.

### Measuring variation

One of the most important expectations is the <span class="term">[variance](https://en.wikipedia.org/wiki/Variance)</span>, which measures the typical squared distance from the mean:

$$ \operatorname{var}(Y) = E[(Y-E[Y])^2]. $$

Another way to write this is

$$ \operatorname{var}(Y) = E[Y^2] - (E[Y])^2. $$

<div class="example" markdown="1">
#### Example (mean and variance of a Bernoulli random variable)

Let $Y \sim \text{Bernoulli}(q)$, with $Y=1$ with probability $q$.

<u>Question:</u> what are $E[Y]$ and $\operatorname{var}(Y)$?

<u>Solution:</u>

$$ E[Y] = P(Y=0)\times 0 + P(Y=1)\times 1 = q. $$

Similarly, you should be able to show $\operatorname{var}(Y) = q(1-q)$; you will also check this formula by simulation in a problem.
</div>

Based on this example, we can estimate $q$ using $\hat q = \bar Y$, as expected &mdash; here $\hat q$ is shorthand for "an estimator of $q$." (We will dive deeper into the concept of an estimator in Unit 3).

To measure "how much variation" there is in a random variable, we'd like to compare the variance to the mean. But there's a problem: the variance has different units than the mean. If mean human height is about $170\,\text{cm}$, the variance might be around $100\,\text{cm}^2$ &mdash; hard to interpret, since it's in squared centimeters. Taking the square root of the variance gives the <span class="term">[standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)</span>, which brings the spread back into the same units as the mean (here, $10\,\text{cm}$).

But even the standard deviation isn't enough when comparing variability across different contexts. Suppose weight has mean $70\,\text{kg}$ and standard deviation $10\,\text{kg}$. The "$10$" here isn't directly comparable to the "$10\,\text{cm}$" for height, because the scales are different &mdash; what matters is variation *relative to the mean*.

This leads to the <span class="term">[coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) (CV)</span>,

$$ \text{CV} = \frac{\sigma}{\mu}, $$

where $\sigma$ is the standard deviation and $\mu$ is the mean. The CV is unitless, so it allows comparisons across variables measured in different units or with very different scales. For example, a CV of $0.06$ in height ($10/170$) indicates less relative variability than a CV of $0.14$ in weight ($10/70$).

| Quantity | Typical CV |
|---|---|
| Adult height | $\approx 0.05$ |
| Adult weight | $\approx 0.2$ |
| Household income | $\approx 1$ |
| Household wealth | $3$&ndash;$5$ |

### Conditional expectation

The <span class="term">[conditional expectation](https://en.wikipedia.org/wiki/Conditional_expectation)</span> is the expectation of the conditional distribution:

$$ E[X \mid Y=y] = \sum_x x\,P(X=x\mid Y=y). $$

With samples $\lbrace (x_1,y_1),\dots,(x_n,y_n) \rbrace$ and at least one observation with $Y=y$,

$$ E[X\mid Y=y] \approx \frac{1}{N(Y=y)}\sum_{i=1}^n 1_{\lbrace y_i=y \rbrace}\,x_i, $$

where $1_{\lbrace y_i=y \rbrace}$ is the <span class="term">[indicator function](https://en.wikipedia.org/wiki/Indicator_function)</span> (1 if $y_i=y$, 0 otherwise). Put simply: we compute a conditional expectation from data by taking the sample average among samples that satisfy the condition.

It should be clear that if $X$ and $Y$ are independent, then $E[X\mid Y=y]=E[X]$ for every $y$ with $P(Y=y)>0$ &mdash; but **the converse is false: it's possible for this to hold even when $X$ and $Y$ are not independent!**

<div class="example" markdown="1">
#### Example ($E[X\mid Y]=E[X]$ without independence)

Let $Y \in \lbrace -1,1\rbrace$ with $P(Y=1)=P(Y=-1)=1/2$. Given $Y=1$, let $X\in\lbrace -1,1\rbrace$ each with probability $1/2$; given $Y=-1$, let $X\in\lbrace -2,2\rbrace$ each with probability $1/2$.

<u>Question:</u> compute $E[X\mid Y=1]$ and $E[X\mid Y=-1]$. Are $X$ and $Y$ independent?

<u>Solution:</u>

$$ E[X\mid Y=1] = \tfrac12(-1)+\tfrac12(1) = 0, \qquad E[X\mid Y=-1] = \tfrac12(-2)+\tfrac12(2) = 0. $$

So $E[X\mid Y=y]=0=E[X]$ for both values of $y$ &mdash; the conditional expectation doesn't depend on $y$ at all. But $X$ and $Y$ are **not** independent: for instance $P(X=1\mid Y=1)=\tfrac12$ while $P(X=1\mid Y=-1)=0$, since $X$ can't even equal $1$ when $Y=-1$. Matching conditional means only tells us the *average* of $X$ doesn't shift with $Y$; the whole *distribution* of $X$ can still change with $Y$.
</div>

<div class="example" markdown="1">
#### Example (computing a conditional expectation)

Consider $(Y_A,Y_B)$ with the joint distribution from Unit 1:

$$ P(Y_A=y_A,Y_B=y_B) = \begin{cases} 1/2 & y_A=0,\,y_B=0 \\ 1/8 & y_A=0,\,y_B=1 \\ 1/8 & y_A=1,\,y_B=0 \\ 1/4 & y_A=1,\,y_B=1 \end{cases} $$

<u>Question:</u> compute $E[Y_A \mid Y_B=1]$.

<u>Solution:</u>

$$ P(Y_A=1\mid Y_B=1) = \frac{P(Y_A=1,Y_B=1)}{P(Y_B=1)} = \frac{1/4}{3/8} = \frac23, $$

so $P(Y_A=0\mid Y_B=1)=1/3$ and $Y_A\mid(Y_B=1) \sim \text{Bernoulli}(2/3)$, which means

$$ E[Y_A\mid Y_B=1] = \frac23. $$

</div>

For the small dataset in the demo below,

$$ \lbrace (Y_1,Y_2)\rbrace = \lbrace (1,2),(1,2),(3,1),(1,4),(3,3),(2,2),(1,5) \rbrace. $$

To compute $E[Y_1\mid Y_2=2]$, first keep only the rows with $Y_2=2$:

$$ (1,2),\ (1,2),\ (2,2). $$

Then average the $Y_1$ values in those rows:

$$ E[Y_1\mid Y_2=2]\approx \frac{1+1+2}{3}=\frac43. $$

The demo repeats this same two-step procedure for a few different conditions: pick a conditional average, and it keeps the rows satisfying the condition and highlights the numbers being averaged. Those two steps are exactly the two steps in the formula above &mdash; restrict to the rows meeting the condition, then average the target column over them.

{% include_relative demos/condexp.html %}

<div class="example" id="ex-kidiq" markdown="1">
#### Example (conditional expectation from data)

Consider data on children's test scores. The variable `kid_score` is the child's test score, and `mom_hs` is $1$ if the mother graduated high school and $0$ otherwise.

pandas is used only to load the CSV; the rest of the example works with the numpy arrays `kid_score` and `mom_hs`.

```python
import numpy as np
import pandas as pd

url = (
    "https://raw.githubusercontent.com/"
    "avehtari/ROS-Examples/"
    "master/KidIQ/data/kidiq.csv"
)
df = pd.read_csv(url)
kid_score = df["kid_score"].to_numpy()
mom_hs = df["mom_hs"].to_numpy()
```

Let $Y$ be the test score and let $X$ be `mom_hs`. The conditional expectation $E[Y\mid X=1]$ is the average test score among children whose mothers graduated high school. The conditional expectation $E[Y\mid X=0]$ is the corresponding average among children whose mothers did not.

```python
overall = np.mean(kid_score)
mean_given_hs0 = np.mean(kid_score[mom_hs == 0])
mean_given_hs1 = np.mean(kid_score[mom_hs == 1])

print("E[Y] estimated from all rows:", overall)
print("E[Y | X=0] estimated within group:", mean_given_hs0)
print("E[Y | X=1] estimated within group:", mean_given_hs1)
```

This code is doing the same operation as the definition: it restricts the rows to a condition, then averages $Y$ inside that restricted sample space. If the two conditional averages differ substantially, then $E[Y\mid X=x]$ depends on $x$, which is evidence that $X$ and $Y$ are not independent in this dataset.

</div>

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Computing conditional averages

Suppose we have data on a pair $(Y_1,Y_2)$:

$$ \lbrace (1,2),(1,2),(3,1),(1,4),(3,3),(2,2),(1,5) \rbrace. $$

Compute each quantity by hand. Then describe, in one sentence for each part, exactly which rows are being averaged.

<ol type="a">
  <li>$E[Y_1]$</li>
  <li>$E[Y_1 \mid Y_2=2]$</li>
  <li>$E[Y_2 \mid Y_1=1]$</li>
  <li>$E[Y_2 \mid Y_1>1]$</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Conditional means from a joint table

Suppose $(X,Y)$ has joint distribution

<table class="prob-table">
<tr><th></th><th>$X=0$</th><th>$X=1$</th><th>$X=2$</th></tr>
<tr><th>$Y=0$</th><td>0.10</td><td>0.20</td><td>0.10</td></tr>
<tr><th>$Y=1$</th><td>0.15</td><td>0.15</td><td>0.30</td></tr>
</table>

Compute $E[X]$, $E[X\mid Y=0]$, and $E[X\mid Y=1]$.
</div>

</details>

### Properties of expectation

Expectation has several important properties, which become especially relevant once we work with linear regression models (defined in terms of conditional expectations).

<ol>
<li><strong>Linearity:</strong> for random variables $X$ and $Y$, $E[X+Y] = E[X]+E[Y]$.

<details markdown="1">
<summary>Proof</summary>

For discrete $S_X,S_Y$:

$$ E[X+Y] = \sum_{y \in S_Y}\sum_{x\in S_X} (x+y)P(X=x,Y=y) $$

$$ = \sum_{x} x\Big(\sum_{y}P(X=x,Y=y)\Big) + \sum_{y} y\Big(\sum_{x} P(X=x,Y=y)\Big) $$

$$ = E[X]+E[Y]. $$

</details>
</li>
<li><strong>Multiplication by a constant:</strong> if $a$ is a constant (not random), $E[aX] = aE[X]$.</li>
<li><strong>Factoring for independent variables:</strong> if $X$ and $Y$ are independent, $E[XY]=E[X]E[Y]$.

<details markdown="1">
<summary>Proof</summary>

Using independence,

$$ E[XY] = \sum_{x}\sum_{y} xy\,P(X=x,Y=y) = \sum_x\sum_y xP(X=x)\,yP(Y=y) $$

$$ = \Big(\sum_x xP(X=x)\Big)\Big(\sum_y yP(Y=y)\Big) = E[X]E[Y]. $$

</details>
</li>
<li><strong>Tower property:</strong> for random variables $X$ and $Y$, $E[E[X\mid Y]] = E[X]$, where $E[X\mid Y]$ is the random variable obtained by applying the deterministic function $f(y)=E[X\mid Y=y]$ to $Y$; that is, $E[X\mid Y] = f(Y)$, so $E[E[X\mid Y]] = E[f(Y)]$.</li>
<li><strong>Law of total variance:</strong> variance has an analogous decomposition,

$$ \operatorname{var}(X) = \underbrace{E[\operatorname{var}(X\mid Y)]}_{\text{within-group variance}} + \underbrace{\operatorname{var}(E[X\mid Y])}_{\text{between-group variance}}. $$

The total variance splits into the average spread *within* each group, plus how much the group *means* themselves vary.</li>
</ol>

<div class="example" markdown="1">
#### Example (calculating conditional expectations with the tower property)

Let $X$ indicate whether a student is in a morning section ($X=0$) or afternoon section ($X=1$). Suppose

$$ P(X=0)=0.6,\qquad P(X=1)=0.4, $$

and suppose quiz scores satisfy

$$ E[Y\mid X=0]=78,\qquad E[Y\mid X=1]=86. $$

<u>Question:</u> compute $E[Y]$ using the tower property, both by hand and with code.

<u>Solution:</u> by the tower property,

$$ E[Y] = E[\,E[Y\mid X]\,] $$

$$ = E[Y\mid X=0]P(X=0)+E[Y\mid X=1]P(X=1) $$

$$ = 78(0.6)+86(0.4)=81.2. $$

The overall mean is a weighted average of the conditional means, with weights given by the group probabilities. The code below confirms the tower property numerically.

```python
import numpy as np

rng = np.random.default_rng(2025)

N = 100_000
X = rng.choice([0, 1], size=N, p=[0.6, 0.4])
Y = np.empty(N)
Y[X == 0] = rng.normal(loc=78, scale=8, size=np.sum(X == 0))
Y[X == 1] = rng.normal(loc=86, scale=8, size=np.sum(X == 1))

EY_given_X0_hat = np.mean(Y[X == 0])
EY_given_X1_hat = np.mean(Y[X == 1])
pX0_hat = np.mean(X == 0)
pX1_hat = np.mean(X == 1)
EEY_given_X_hat = EY_given_X0_hat * pX0_hat + EY_given_X1_hat * pX1_hat

print(f"Theoretical E[Y]    = 81.200000")
print(f"Estimated E[Y]      = {np.mean(Y):.6f}")
print(f"Estimated E[E[Y|X]] = {EEY_given_X_hat:.6f}")
```
</div>

<div class="example" markdown="1">
#### Example (law of total variance)

Continuing the previous example, suppose additionally that quiz scores have the same spread in each section, $\operatorname{var}(Y\mid X=0)=\operatorname{var}(Y\mid X=1)=64$.

<u>Question:</u> compute $\operatorname{var}(Y)$ using the law of total variance.

<u>Solution:</u> the within-group term is the weighted average of the (here, equal) conditional variances:

$$ E[\operatorname{var}(Y\mid X)] = 0.6(64)+0.4(64) = 64. $$

The between-group term is the variance of the conditional means $78$ and $86$ around the overall mean $E[Y]=81.2$, weighted by $P(X=0)=0.6$ and $P(X=1)=0.4$:

$$ \operatorname{var}(E[Y\mid X]) = 0.6(78-81.2)^2+0.4(86-81.2)^2 = 6.144+9.216=15.36. $$

So

$$ \operatorname{var}(Y) = 64+15.36 = 79.36. $$

Even though every section has the *same* internal spread, the overall variance is bigger than either conditional variance, because the two sections also have different means. The code below extends the previous simulation to confirm this.

```python
within_hat = pX0_hat * np.var(Y[X == 0]) + pX1_hat * np.var(Y[X == 1])
between_hat = (
    pX0_hat * (EY_given_X0_hat - np.mean(Y)) ** 2
    + pX1_hat * (EY_given_X1_hat - np.mean(Y)) ** 2
)

print(f"Theoretical var(Y)                   = 79.360000")
print(f"Estimated var(Y)                     = {np.var(Y):.6f}")
print(f"Estimated E[var(Y|X)] + var(E[Y|X])  = {within_hat + between_hat:.6f}")
```
</div>

<div class="example" markdown="1">
#### Example (expectation of a sum)

Let $Y = \sum_{i=1}^N X_i$ with $X_i \sim \text{Bernoulli}(q)$ iid. If you read the optional section in Unit 1, you will recognize that this means $Y \sim \text{Binomial}(N,q)$.

<u>Question:</u> what are $E[Y]$ and $\operatorname{var}(Y)$?

<u>Solution:</u> by linearity (property 1),

$$ E[Y] = E\Big[\sum_{i=1}^N X_i\Big] \underset{(1)}{=} \sum_{i=1}^N E[X_i] = Nq. $$

Similarly,

$$ E[Y^2] = E\Big[\Big(\sum_i X_i\Big)^2\Big] = E\Big[\sum_i\sum_j X_iX_j\Big] \underset{(1)}{=} \sum_i\sum_j E[X_iX_j] $$

$$ \underset{(3)}{=} \sum_{i}\sum_{j\ne i} q^2 + \sum_i E[X_i^2] = N(N-1)q^2 + Nq, $$

using $E[X_i^2]=E[X_i]=q$ and property 3 (factoring) for the $i\ne j$ terms. Therefore

$$ \operatorname{var}(Y) = E[Y^2]-E[Y]^2 = N(N-1)q^2+Nq - N^2q^2 = Nq(1-q). $$

</div>

To summarize:

$$ E[Y] = qN, \qquad \operatorname{var}(Y) = Nq(1-q). $$

Both the mean and variance grow linearly with $N$, but the standard deviation only grows like $\sqrt N$. Relative to the mean, the spread shrinks, which is captured by the CV:

$$ \text{CV} = \frac{\sqrt{\operatorname{var}(Y)}}{E[Y]} = \sqrt{\frac{1-q}{q}\cdot\frac1N}. $$

This measures variation *relative* to the average, which matters for many applications.

<div class="example" markdown="1">
#### Example (election modeling)

Consider an election between two candidates. Let $q$ be the population fraction supporting candidate one, and suppose $N$ people vote (with $N$ much smaller than the population, since turnout is low). Each voter's ballot is a Bernoulli($q$) indicator of support for candidate one, so the number of votes for candidate one, $M$, is a sum of $N$ iid Bernoulli($q$) draws, i.e. $M \sim \text{Binomial}(N,q)$, with mean and variance as derived above.

<u>Question:</u> in a city where $q=0.51$ support a candidate, and $N=1000$ vote, what's the chance the vote share $\hat q=M/N$ differs from $q$ by more than $1\%$? There's no closed form for this, so we answer by simulation.

```python
import numpy as np

q = 0.51
N = 1000
delta = 0.01
trials = 200_000

rng = np.random.default_rng(123)

M_samples = rng.binomial(n=N, p=q, size=trials)
q_hat = M_samples / N
prob_est = np.mean(np.abs(q_hat - q) > delta)

print(f"Monte Carlo estimate P(|q_hat - {q}| > {delta}) = {prob_est:.6f}")
```
</div>

In this problem, the vote share is exactly the sample mean of the individual votes, $\hat q = M/N$, so $E[\hat q]=q$. The variance is

$$ \operatorname{var}(\hat q) = \operatorname{var}(M/N) = \frac{1}{N^2}\operatorname{var}(M) = \frac{q(1-q)}{N}, $$

which tends to zero as $N\to\infty$, while $E[\hat q]$ doesn't depend on $N$ at all. This is a consequence of the CV decreasing with $N$, and it's what lets us determine $q$ by approximating $E[\hat q]$ with the sample mean.

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Expectation and variance from a table

Let $Y$ have distribution

$$ P(Y=0)=0.2,\qquad P(Y=1)=0.5,\qquad P(Y=2)=0.3. $$

Compute $E[Y]$, $E[Y^2]$, and $\operatorname{var}(Y)$.
</div>

<div class="exercise" markdown="1">
#### Weighted conditional means

Suppose $P(X=0)=0.7$, $P(X=1)=0.3$, $E[Y\mid X=0]=10$, and $E[Y\mid X=1]=20$. Use the tower property to compute $E[Y]$.
</div>

<div class="exercise" markdown="1">
#### Law of total variance from conditional moments

Suppose $P(X=0)=0.7$, $P(X=1)=0.3$, $E[Y\mid X=0]=10$, $E[Y\mid X=1]=20$, $\operatorname{var}(Y\mid X=0)=4$, and $\operatorname{var}(Y\mid X=1)=9$. Use the law of total variance to compute $\operatorname{var}(Y)$.
</div>

<div class="exercise" markdown="1">
#### Sum of Bernoulli variables

Let $Y=\sum_{i=1}^{12}X_i$, where the $X_i$ are iid Bernoulli$(0.25)$. Compute $E[Y]$, $\operatorname{var}(Y)$, and the coefficient of variation.
</div>

</details>

## 2.2 The Normal distribution {#sec-2-2}

We now define the most important distribution in statistics: the <span class="term">[Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)</span>. Recall from [Unit 1](../unit1/#sec-1-4) that a continuous random variable $Y$ is characterized by a *density* $f(y)$, with $P(a<Y<b)=\int_a^b f(y)\,dy$. The Normal distribution is continuous, symmetric, bell-shaped, and completely determined by two parameters: its mean $\mu$ and variance $\sigma^2$. Its density is

$$ g(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}. $$

Despite the simplicity of this formula, calculating Normal probabilities by hand (integrating to find area under the curve) is difficult. Instead we rely on rough rules of thumb: about $68\%$ of the probability lies within $1$ standard deviation of the mean, about $95\%$ lies within $2$ standard deviations, and about $99.7\%$ lies within $3$ standard deviations. The demo below keeps the $x$-axis fixed while you change $\mu$ and $\sigma$, so you can see that $\mu$ shifts the center and $\sigma$ changes the spread. Its two shaded bands are $\mu\pm\sigma$ and $\mu\pm2\sigma$, and they hold about $68\%$ and $95\%$ of the area wherever you put the sliders &mdash; a direct consequence of the Normal distribution's bell curve shape.

If $X$ has density $g(x)$ above, we write $X \sim \text{Normal}(\mu,\sigma^2)$. It can be shown that $E[X]=\mu$ and $\operatorname{var}(X)=\sigma^2$ &mdash; hence the name. **Be careful:** sometimes (in code or in math) Normal random variables are parameterized by mean and standard deviation instead of mean and variance, so you'll also see $\text{Normal}(\mu,\sigma)$. Both conventions are used, so always check.

In code, `np.random.normal(mu, sigma)` (or, with a `Generator`, `rng.normal(mu, sigma)`) draws a sample from $\text{Normal}(\mu,\sigma^2)$ &mdash; note it takes the standard deviation `sigma`, not the variance.

{% include_relative demos/normal.html %}

<div class="example" markdown="1">
#### Example (calculating Normal probabilities)

Suppose $Y \sim \text{Normal}(5,4)$.

<u>Question:</u> what is (approximately) $P(Y>7)$?

<u>Solution:</u> since $5+2=7$, this is asking how likely a Normal variable is to exceed $1$ standard deviation above the mean &mdash; about $16\%$. (We can always compute this exactly in Python too.)

<u>Question:</u> what is $P(Y>3\mid Y<7)$?

<u>Solution:</u>

$$ P(Y>3\mid Y<7) = \frac{P(Y>3,Y<7)}{P(Y<7)}. $$

Since $3=5-2=\mu-\sigma$ and $7=5+2=\mu+\sigma$, we have $P(Y<7)\approx0.841$ and $P(Y>3,Y<7)\approx0.682$, so the answer is about $0.81$.
</div>

### Properties of Normal random variables

**Linear transformations of Normal random variables.** Suppose $Z \sim \text{Normal}(0,1)$ and define $X = \sigma Z + \mu$. Write $\phi(z) = e^{-z^2/2}/\sqrt{2\pi}$ for the density of $Z$ &mdash; this is the density $g$ above with $\mu=0,\sigma=1$. Then

$$ P(X<x) = P(Z < \tfrac{x-\mu}{\sigma}) = \int_{-\infty}^{(x-\mu)/\sigma} \phi(z)\,dz. $$

Substituting $u=\mu+\sigma z$ (so $dz=du/\sigma$), when $z\to-\infty$ we get $u\to-\infty$, and at $z=(x-\mu)/\sigma$ we get $u=x$:

$$ P(X<x) = \int_{-\infty}^x \phi\!\left(\frac{u-\mu}{\sigma}\right)\frac1\sigma\,du = \int_{-\infty}^x \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(u-\mu)^2}{2\sigma^2}\right) du. $$

This shows $X \sim \text{Normal}(\mu,\sigma^2)$ &mdash; in particular, any Normal random variable is a linear transformation of a standard normal. More generally, if $X \sim \text{Normal}(\mu_x,\sigma_x^2)$ and $Y=aX+b$, then $Y$ should also be Normal. Taking expectations, $E[Y]=a\mu_x+b$, and since $\operatorname{var}(aX)=a^2\operatorname{var}(X)$ and $\operatorname{var}(b)=0$,

$$ Y \sim \text{Normal}(a\mu_x+b,\ a^2\sigma_x^2). $$

Going from $Z$ to $X$ to $Y$, we're just multiplying and shifting &mdash; think about what that does to a histogram. The process of going from $X$ to $Z$ is called <span class="term">[standardizing](https://en.wikipedia.org/wiki/Standard_score)</span>: for any $X$, the standardized variable is

$$ Z = \frac{X-\mu_x}{\sigma_x}. $$

Transforming $X$ to a standard normal is equivalent to measuring $X$ in units of standard deviations &mdash; if you make a histogram of $X$, this transformation just relabels the $x$-axis in those units.

**Theorem (sums of independent Normal random variables).** If $X_1 \sim \text{Normal}(\mu_1,\sigma_1^2)$ and $X_2 \sim \text{Normal}(\mu_2,\sigma_2^2)$ are independent, then

$$ aX_1+bX_2+d \sim \text{Normal}\big(a\mu_1+b\mu_2+d,\ a^2\sigma_1^2+b^2\sigma_2^2\big). $$

We'll use this in Unit 3, where via the Central Limit Theorem, we can approximate the distribution of almost any sample average with a Normal distribution.

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Normal probabilities

Suppose $Y\sim\operatorname{Normal}(10,9)$.

<ol type="a">
  <li>Write the standardized version of $Y$.</li>
  <li>Estimate $P(Y>13)$ using the bell curve shape.</li>
  <li>Estimate $P(7<Y<13)$ using the bell curve shape.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Linear transformation

Suppose $X \sim \operatorname{Normal}(5,9)$ and define $Y=-2X+7$.

<ol type="a">
  <li>Find the distribution of $Y$ by giving its mean and variance.</li>
  <li>Standardize $Y$, then use the bell curve shape to estimate $P(Y>3)$.</li>
  <li>Without computing anything, write down $P(Y>-3)$ and explain in one sentence how you know.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Sum of Normal random variables

Suppose $X_1\sim\operatorname{Normal}(2,4)$ and $X_2\sim\operatorname{Normal}(-1,9)$ are independent, and define $S=3X_1-2X_2$.

<ol type="a">
  <li>Find the distribution of $S$ by giving its mean and variance.</li>
  <li>Standardize $S$.</li>
  <li>Use the bell curve shape to estimate $P(0<S<16)$.</li>
</ol>
</div>

</details>

## 2.3 Linear regression: a conditionally Normal model {#sec-2-3}

Equipped with conditional expectation and the Normal distribution, we can define our first regression model. A (single-predictor) <span class="term">[linear regression model](https://en.wikipedia.org/wiki/Linear_regression)</span> is conditionally Normal:

$$ Y\mid X \sim \text{Normal}(\beta_0+\beta_1X,\ \sigma^2), $$

where $X$ is the <span class="term">[predictor](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)</span> and $Y$ is the <span class="term">[response variable](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)</span>.

The conditional expectation is the line

$$ E[Y\mid X]=\beta_0+\beta_1X, $$

and the Normal distribution describes the vertical spread of $Y$ around that line. Equivalently, we can write

$$ Y=\beta_0+\beta_1X+\epsilon,\qquad \epsilon\sim\text{Normal}(0,\sigma^2), $$

where $\epsilon$ is the noise or error term. Let us summarize the assumptions being made in this model:

<ol type="a">
  <li><strong>Linearity:</strong> the conditional mean $E[Y\mid X]=\beta_0+\beta_1X$ is a straight line in $X$.</li>
  <li><strong>Homoskedasticity:</strong> the conditional variance $\operatorname{var}(Y\mid X)=\sigma^2$ is the same at every value of $X$. </li>
  <li><strong>Conditional normality:</strong> $Y\mid X$ (equivalently $\epsilon$) is Normally distributed. </li>
  <li><strong>Exogeneity of $\epsilon$:</strong> $E[\epsilon\mid X]=0$, so after accounting for $X$, the leftover error has mean zero at every value of $X$. A predictor satisfying this is called <span class="term"><a href="https://en.wikipedia.org/wiki/Exogenous_and_endogenous_variables">exogenous</a></span>; if it fails, $X$ is <span class="term"><a href="https://en.wikipedia.org/wiki/Exogenous_and_endogenous_variables">endogenous</a></span>, and $\beta_1$ can no longer be interpreted as the effect of $X$ alone.</li>
</ol>

For example, suppose $X$ indicates whether a student attended a study-skills workshop and $Y$ is their exam score. If $X$ is assigned by a coin flip (a randomized trial), then whatever unmeasured factors drive $\epsilon$ are, on average, balanced between the two groups and as a result $E[\epsilon\mid X]=0$.  But if students instead self-select into the workshop, the more motivated or more anxious students may be the ones who choose to attend, so unmeasured motivation is now correlated with $X$, making $X$ endogenous. 

<div class="example" id="ex-diffmeans" markdown="1">
#### Example (binary predictor: the difference of means)

Let $X \sim \text{Bernoulli}(1/2)$ and $Y\mid X \sim \text{Normal}(\beta_1X+\beta_0,\sigma^2)$ &mdash; motivated by a clinical trial where participants are placed in control ($X=0$) or treatment ($X=1$) groups, and $Y$ is some measured outcome (e.g. blood pressure).

Writing the conditional expectation at the two possible values of $X$ gives

$$ E[Y\mid X=0]=\beta_0,\qquad E[Y\mid X=1]=\beta_0+\beta_1. $$

Therefore

$$ E[Y\mid X=1] - E[Y\mid X=0] = \beta_1. $$

For a binary predictor, the slope $\beta_1$ is the difference between the treatment-group mean and the control-group mean. In data, the natural estimate is the same comparison with conditional sample averages:

$$ \widehat\beta_1 = \overline{Y\mid X=1} - \overline{Y\mid X=0}, $$

$$ \overline{Y\mid X=1} = \frac{1}{N(X=1)}\sum_{i=1}^n Y_i 1_{X_i=1}, $$

where $n$ is the number of samples and $N(X=1)$ counts those with $X_i=1$.

The functions below implement these expressions in code.

```python
import numpy as np

def generate_data(beta0, beta1, sigma, n_samples):
    x = np.random.choice([0, 1], n_samples)
    y = beta0 + beta1 * x + np.random.normal(0, sigma, n_samples)
    return x, y

def beta1_hat(x, y):
    return y[x == 1].mean() - y[x == 0].mean()
```
</div>

<div class="example" markdown="1">
#### Example (linear regression with a Normal predictor)

Let $X \sim \text{Normal}(\mu_x,\sigma_x^2)$ and $Y\mid X \sim \text{Normal}(\beta_1X+\beta_0,\sigma^2)$. Now we can't estimate $\beta_1$ by comparing means at $X=0$ vs. $X=1$, since we may never observe those exact values &mdash; we'll need another approach, leading to least squares.

<u>Question:</u> what's the marginal distribution of $Y$? What is $E[XY]$, and how does it compare to $E[X]E[Y]$?

<u>Solution:</u> write $Y=\beta_1X+\beta_0+Z$ with $Z\sim\text{Normal}(0,\sigma^2)$ independent of $X$. Then the marginal distribution of $Y$ is a sum of two independent Normals, so

$$ Y \sim \text{Normal}(\beta_1\mu_x+\beta_0,\ \beta_1^2\sigma_x^2+\sigma^2). $$

For $E[XY]$: since $E[XY\mid X=x] = xE[Y\mid X=x]$,

$$ E[XY] = E[X\,E[Y\mid X]] = E[X(\beta_1X+\beta_0)] = \beta_1E[X^2]+\beta_0E[X]. $$

Using $E[X^2]=\operatorname{var}(X)+E[X]^2=\sigma_x^2+\mu_x^2$,

$$ E[XY] = \beta_1\sigma_x^2+\beta_1\mu_x^2+\beta_0\mu_x. $$

Meanwhile $E[X]E[Y] = \mu_x(\beta_1\mu_x+\beta_0) = \beta_1\mu_x^2+\beta_0\mu_x$. The difference between the two is the extra term $\beta_1\sigma_x^2$, which comes from the variance of $X$.
</div>

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Binary predictor means

Suppose $X\in\lbrace 0,1\rbrace$ and

$$ Y\mid X \sim \operatorname{Normal}(\beta_0+\beta_1X,\sigma^2). $$

Compute $E[Y\mid X=0]$, $E[Y\mid X=1]$, and $E[Y\mid X=1]-E[Y\mid X=0]$. Then explain why the difference of conditional sample averages is a natural estimator of $\beta_1$.
</div>

</details>

## Problems {#problems}

<div class="exercise" markdown="1">
#### Problem 2.1 &mdash; Checking the Bernoulli variance formula

Write Python code that checks the formula $\operatorname{var}(Y)=q(1-q)$ for $Y\sim\operatorname{Bernoulli}(q)$.

Your final answer should include:

<ol type="a">
  <li>a plot comparing empirical variance to $q(1-q)$ for many values of $q$;</li>
  <li>a sentence explaining what dataset was simulated;</li>
  <li>a sentence explaining why the plot supports the formula but does not prove it.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 2.2 &mdash; Conditional expectations in the KidIQ data

Return to the KidIQ dataset from [Section 2.1](#ex-kidiq).

Your analysis should answer:

<ol type="a">
  <li>What are the sample estimates of $E[Y]$, $E[Y\mid X=1]$, and $E[Y\mid X=0]$, where $Y$ is `kid_score` and $X$ is `mom_hs`?</li>
  <li>Does the difference in conditional averages suggest independence or dependence between `kid_score` and `mom_hs`?</li>
  <li>Now compare the average value of `mom_iq` in the two `mom_hs` groups. Why might this make $X$ endogenous in a regression of `kid_score` on `mom_hs` alone?</li>
  <li>Write two sentences separating the associational claim supported by these computations from the stronger causal claim they do not establish.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 2.3 &mdash; Simulating a binary-predictor regression model

Write Python code for the model from [Section 2.3](#ex-diffmeans):

$$ X\sim\operatorname{Bernoulli}(1/2), \qquad
Y\mid X\sim\operatorname{Normal}(\beta_0+\beta_1X,\sigma^2). $$

Choose concrete values for $\beta_0$, $\beta_1$, and $\sigma$, then simulate one dataset.

Your final answer should include:

<ol type="a">
  <li>the simulated conditional averages $\overline{Y\mid X=0}$ and $\overline{Y\mid X=1}$;</li>
  <li>the difference $\overline{Y\mid X=1}-\overline{Y\mid X=0}$ and the true value of $\beta_1$;</li>
  <li>a plot comparing the simulated $Y$ values in the two groups;</li>
  <li>two sentences explaining what $\beta_0$, $\beta_1$, and $\sigma$ control in this model.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 2.4 &mdash; Checking the law of total variance by simulation

Let $X$ indicate group membership with $P(X=0)=P(X=1)=0.5$, and suppose

$$ Y\mid X=0 \sim \operatorname{Normal}(50,10^2), \qquad Y\mid X=1 \sim \operatorname{Normal}(70,20^2). $$

Note that, unlike the worked example in [Section 2.1](#sec-2-1), the two conditional variances are **not** equal here.

Write Python code that:

<ol type="a">
  <li>simulates $N=100{,}000$ samples of $(X,Y)$ from this model;</li>
  <li>estimates $E[\operatorname{var}(Y\mid X)]$ (within-group) and $\operatorname{var}(E[Y\mid X])$ (between-group) from the simulated data, and adds them together;</li>
  <li>compares that sum to the sample variance of all of $Y$ taken together, and confirms they closely agree;</li>
  <li>by hand, computes the exact value of $\operatorname{var}(Y)$ using the law of total variance, and states how closely it matches part (c).</li>
</ol>
</div>
