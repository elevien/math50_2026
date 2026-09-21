---
layout: page
title: Exam study guide
---

There are **two midterms** this term, each taking half of a class period, plus a cumulative final. See schedule.

---

## Midterm 1

Covers [Unit 1]({{ '/unit1/' | relative_url }}) and [Unit 2]({{ '/unit2/' | relative_url }}) &mdash; everything through Section 2.4.

### What you need to know

**Unit 1**

- The only named distributions you need to memorize are **Bernoulli**, **Uniform** and **Normal**.
- Sample spaces, outcomes and events; the three axioms and how to use them.
- Read a joint distribution off a table and compute marginal, joint and conditional probabilities from it. Determine if variables are independent. 
- Go from a model written in $\sim$ notation, such as $Y \sim \text{Bernoulli}(q)$ and $X \mid Y \sim \text{Bernoulli}(Y/4 + 1/4)$, to the joint distribution, and back.
- Read and produce the output of simple Python code (arrays, for loops, if statements, `np.random`). Emphasis will be placed on translating simulations to probability statements.

**Unit 2**

- Know propeties of expectation and compute $E[Y]$, $E[g(Y)]$, $\operatorname{var}(Y)$ and the coefficient of variation for a discrete random variable, by hand as well as conditional versions. 
- Use the tower property and the law of total variance to compute $E[Y]$ and $\operatorname{var}(Y)$ from conditional moments.
- Standardize a Normal variable, and use the bell curve shape (68 / 95 / 99.7) to estimate probabilities.
- The distribution of $aX+b$, and of a sum of independent Normals.
- The single-predictor model $Y \mid X \sim \text{Normal}(\beta_0+\beta_1X, \sigma^2)$, its equivalent form $Y = \beta_0+\beta_1X+\epsilon$, and what each parameter controls.
- For a **binary** predictor, why $\beta_1 = E[Y\mid X{=}1]-E[Y\mid X{=}0]$, and why the difference of group averages estimates it.
- What it means for a predictor to be **exogenous** or **endogenous**, and why endogeneity makes $\beta_1$ hard to interpret. 
- $\operatorname{cov}(X,Y)=\beta_1\sigma_X^2$ for a general (not necessarily binary) predictor, and the resulting formula for $\hat\beta_1$ from data.
- The correlation $\rho = \operatorname{cov}(X,Y)/(\sigma_X\sigma_Y) = \beta_1\sigma_X/\sigma_Y$, and **regression to the mean**: why an unusually extreme standardized $X$ predicts a less extreme standardized $Y$ whenever $\lvert\rho\rvert<1$.
- **Coefficient of determination**: the relationships $\sigma_Y^2 = \beta_1^2\sigma_X^2+\sigma_\epsilon^2$ and $\rho^2 = 1-\sigma_\epsilon^2/\sigma_Y^2$, and its sample estimate $R^2$. You should be able to derive these, not just quote them.


### Practice problems

<div class="exercise" markdown="1">
#### Problem 1

$(X,Y)$ has $X\in\lbrace0,1\rbrace$, $Y\in\lbrace0,1,2\rbrace$ and joint distribution

| $P(X,Y)$ | $Y=0$ | $Y=1$ | $Y=2$ |
|---|---|---|---|
| $X=0$ | 0.10 | 0.15 | 0.20 |
| $X=1$ | 0.25 | 0.05 | 0.25 |

1. Compute $P(X=1)$ and $P(Y=2)$.
1. Compute $P(X=1\mid Y=2)$.
1. Are $X$ and $Y$ independent? Justify with a single calculation.
1. Compute $E[Y\mid X=0]$.
</div>

<div class="exercise" markdown="1">
#### Problem 2

Let $Y \sim \text{Bernoulli}(1/3)$ and $X \mid Y \sim \text{Bernoulli}(1/4 + Y/2)$.

1. Write out the full joint distribution of $(X,Y)$ as a $2\times2$ table, and check that it sums to 1.
1. Compute the marginal $P(X=1)$.
1. Compute $P(Y=1\mid X=1)$.
1. Compute $E[X]$ two ways: from the marginal in (b), and using the tower property.
</div>

<div class="exercise" markdown="1">
#### Problem 3

$Z$ takes the value $-1$ with probability $0.2$, $0$ with probability $0.5$ and $2$ with probability $0.3$.

1. Compute $E[Z]$ and $E[Z^2]$.
1. Compute $\operatorname{var}(Z)$ and the standard deviation.
1. Compute the coefficient of variation. Why is it so large here, and what would happen to it if every value of $Z$ were shifted up by 10?
</div>

<div class="exercise" markdown="1">
#### Problem 4

Quiz scores $Y$ depend on which section a student is in. $45\%$ of students are in the morning section, where $E[Y\mid\text{morning}]=72$; the rest are in the afternoon section, where $E[Y\mid\text{afternoon}]=80$.

1. Compute $E[Y]$.
1. Suppose instead you are told $E[Y]=76.4$ and $E[Y\mid\text{morning}]=72$, but not the afternoon mean. Recover it.
1. In one sentence, say why $E[Y]$ is *not* the average of $72$ and $80$.
</div>

<div class="exercise" markdown="1">
#### Problem 5

Let $W \sim \text{Normal}(12, 16)$.

1. Write the standardized version of $W$.
1. Estimate $P(W>16)$ and $P(4<W<20)$ using the bell curve shape.
1. Let $V = 3-2W$. Give the distribution of $V$.
1. Let $W_1,W_2$ be independent copies of $W$. Give the distribution of $(W_1+W_2)/2$, and say in one sentence how its spread compares to that of $W$.
</div>

<div class="exercise" markdown="1">
#### Problem 6

```python
n = 100_000
y = np.random.choice([0, 1], size=n, p=[0.4, 0.6])

x = np.empty(n)
x[y == 0] = np.random.normal(-1, 1, len(y[y == 0]))
x[y == 1] = np.random.normal(2, 1, len(y[y == 1]))

print(np.mean(x))
print(np.mean(x[y == 1]))
```

1. Write the probability model being simulated, in $\sim$ notation.
1. What number will the first `print` produce, approximately? Show the calculation.
1. What number will the second `print` produce?
1. What quantity would `np.mean(y[x > 0])` estimate? You do not need to compute it.
</div>

<div class="exercise" markdown="1">
#### Problem 7

Let $Y \sim \text{Uniform}(0,4)$.

1. What is the density $f(y)$?
1. Compute $P(1<Y<2.5)$ and $P(Y=2)$.
1. Compute $P(Y<1 \mid Y<3)$.
1. A classmate says a density can never be bigger than 1 because probabilities are at most 1. Give a one-line counterexample and explain the error.
</div>

<div class="exercise" markdown="1">
#### Problem 8

In a randomized trial, $X=1$ for treatment and $X=0$ for control, with $Y \mid X \sim \text{Normal}(\beta_0+\beta_1X,\ \sigma^2)$. Fifty subjects are assigned to each group; the control-group average outcome is $12.4$ and the treatment-group average is $15.1$.

1. Write $E[Y\mid X=0]$ and $E[Y\mid X=1]$ in terms of $\beta_0,\beta_1$.
1. Give the natural estimates $\hat\beta_0$ and $\hat\beta_1$.
1. If assignment is a fair coin flip, what is $E[Y]$ in terms of the fitted values?
1. Suppose subjects had instead *chosen* their own group. Explain, using the definition of exogeneity, why $\hat\beta_1$ would then be hard to interpret.
</div>

<div class="exercise" markdown="1">
#### Problem 9

Consider the model $X \sim \text{Bernoulli}(1/2)$, $Y \mid X \sim \text{Normal}(1+3X,\ 4)$. One of the following implements it correctly.

```python
# A
x = np.random.choice([0, 1], p=[0.5, 0.5], size=n)
y = np.random.normal(1 + 3*x, 4, n)

# B
x = np.random.choice([0, 1], p=[0.5, 0.5], size=n)
y = np.random.normal(1 + 3*x, 2, n)

# C
x = np.random.normal(0.5, 1, n)
y = np.random.normal(1 + 3*x, 2, n)
```

1. Which one is correct?
1. For each of the other two, say precisely which part of the model it gets wrong.
</div>

<div class="exercise" markdown="1">
#### Problem 10

An experiment tests whether *believing* an activity is exercise improves health, independent of any change in actual physical activity. Housekeeping staff at a single hotel are split into a control group and a treatment group; the treatment group is told that their daily work already meets recommended exercise guidelines, while the control group is told nothing. Everyone keeps doing their normal job at the same hotel. Several weeks later, health measurements (e.g. blood pressure) are taken for both groups and the group averages are compared.

1. The description doesn't say how workers were assigned to the two groups. What is the best way to do this assignment, and why?
1. Suppose that instead, workers were free to choose which group they wanted to join. Using the definition of exogeneity, explain why the difference between the two group averages would then be hard to interpret as the effect of the message itself.
1. Even with the assignment from (a), give one reason that having both groups work at the same hotel could still bias the comparison, and suggest a fix.
</div>

<div class="exercise" markdown="1">
#### Problem 11

Consider the following code.

```python
n = 100_000
x = np.random.choice([0, 1], size=n)
y = np.random.normal(3 + 4*x, 2, n)

result = np.mean(y[x == 1]) - np.mean(y[x == 0])
print(result)
```

1. What value will `result` be close to? Show your reasoning.
1. Write down the model being simulated for `y` in the form $Y\mid X \sim \text{Normal}(\beta_0+\beta_1X,\ \sigma^2)$, giving the values of $\beta_0$, $\beta_1$ and $\sigma^2$.
1. What is `result` an estimate of, in terms of $\beta_0$ and $\beta_1$?
</div>

<div class="exercise" markdown="1">
#### Problem 12

Suppose $X \sim \text{Normal}(\mu_X, 4)$ and $Y \mid X \sim \text{Normal}(\beta_0+3X,\ 64)$.

1. Compute $\operatorname{cov}(X,Y)$.
1. Compute $\operatorname{var}(Y)$.
1. Compute the correlation $\rho$ between $X$ and $Y$.
1. If $X$ is $2$ standard deviations above its mean, what is the predicted standardized value of $Y$? What phenomenon does this illustrate?
</div>


---

## Midterm 2

Covers [Unit 3]({{ '/unit3/' | relative_url }}) and [Unit 4]({{ '/unit4/' | relative_url }}).

Midterm 2 is not cumulative in the sense of having questions focusing on Unit 1 and 2 material, but intermediate steps may involve those topics.  

### What you need to know

**Unit 3**

- What the **law of large numbers** and the **Central Limit Theorem** say, and why they matter for inference. Use the CLT to approximate the distribution of a sum or a sample average.
- **Estimators**: what makes something an estimator, and the difference between an estimator and the parameter it targets.
- **Sample distribution**, and the distinction between a *sample* within a dataset and a *replicate* of the whole dataset.
- **Bias** and **consistency**. Given a simple estimator, decide whether it is unbiased and whether it is consistent, and be able to construct an estimator that is one but not the other.
- **Standard error** $\text{se}(\hat\mu)=\sigma/\sqrt N$, and why we usually substitute $\hat\sigma$.
- **Confidence intervals**: compute a 95% CI, solve for the $n$ needed to hit a target width, and state the correct interpretation &mdash; and why it is *not* "there is a 95% chance $\theta$ lies in this interval."
- Least squares: what RSS is, that $\hat\beta_1,\hat\beta_0$ (from Unit 2's covariance formula) minimize it, and estimating the slope and intercept from data by hand for a small dataset.
- **Sample distribution of $\hat\beta_1,\hat\beta_0$**: treating the $x_i$ as fixed, write $\hat\beta_1$ as a weighted sum $\sum_i w_iY_i$ of the (random) $Y_i$, and show it is **unbiased** ($E[\hat\beta_1]=\beta_1$).

**Unit 4**

- **Study design**: randomized controlled trials versus cohort and other observational designs, what a **confounder** is, and when a regression coefficient may and may not be read causally.
- Interpreting $\beta_i$ as an average difference *with the other predictors held fixed*, and what "controlling for" means.
- The relationship $\beta_1' = \beta_1 + \beta_2\beta_{1,2}$ between the single- and multiple-predictor coefficients, and predicting how adding a predictor moves an existing coefficient.
- **Simpson's paradox**: what it is, and the condition under which the sign flips.
- The covariance matrix $\Sigma$, the system $\Sigma\beta = (\operatorname{cov}(X_1,Y),\operatorname{cov}(X_2,Y))^\top$, and solving it for two predictors.
- Why adding a predictor never decreases $R^2$.
- **Collinearity**: its effect on the joint sample distribution of $(\hat\beta_1,\hat\beta_2)$ and on the standard errors.
- **Categorical predictors**: how many dummy variables, why one category is dropped, and how to interpret coefficients relative to the baseline.
- Reading `statsmodels` output: coefficients, standard errors, $p$-values, confidence intervals and $R^2$.


### Practice problems

<div class="exercise" markdown="1">
#### Problem 1

Let $X_1,\dots,X_{100}$ be iid with $E[X_i]=3$ and $\operatorname{var}(X_i)=4$, and let $S=\sum_{i=1}^{100}X_i$.

1. Compute $E[S]$ and $\operatorname{var}(S)$.
1. Write the approximate distribution of $S$, and say what justifies the approximation.
1. Estimate $P(S>340)$ using the bell curve shape.
1. Give the approximate distribution of $\overline X$, and its standard deviation.
</div>

<div class="exercise" markdown="1">
#### Problem 2

Let $X_1,\dots,X_N$ be iid $\text{Bernoulli}(q)$ and write $Y=\sum_i X_i$. For each estimator, state whether it is unbiased, whether it is consistent, and give its standard error.

1. $\hat q_a = Y/N$
1. $\hat q_b = (Y+1)/(N+2)$
1. $\hat q_c = X_1$

Then: which of these is unbiased but *not* consistent, and which is consistent but *not* unbiased? Explain in one sentence each why that combination is possible.
</div>

<div class="exercise" markdown="1">
#### Problem 3

A quantity is Normally distributed with known $\sigma=6$. You collect $n=36$ samples and observe $\overline X = 21.5$.

1. Compute $\text{se}(\hat\mu)$ and the approximate 95% confidence interval.
1. How many samples would you need for the margin of error to be at most $0.5$?
1. A classmate says "there is a 95% chance the true mean is between 19.5 and 23.5." Say what is wrong with this and give the correct statement.
</div>

<div class="exercise" markdown="1">
#### Problem 4

Fit a single-predictor regression to the four points $(1,1)$, $(2,3)$, $(3,4)$, $(4,6)$.

1. Compute $\hat\beta_1$ and $\hat\beta_0$.
1. Compute the four residuals and the RSS.
1. Compute $R^2$.
1. Now add the point $(20,2)$. Without redoing the calculation, say which direction $\hat\beta_1$ moves and why.
</div>

<div class="exercise" markdown="1">
#### Problem 5

For a single-predictor regression model you are told $\sigma_X = 4$, $\sigma_Y = 10$ and $\operatorname{cov}(X,Y)=24$.

1. Compute the correlation $\rho$.
1. Compute the regression slope $\beta_1$.
1. Compute $\rho^2$ and the noise standard deviation $\sigma_\epsilon$.
1. Verify your answers are consistent with $\sigma_Y^2=\beta_1^2\sigma_X^2+\sigma_\epsilon^2$.
1. If $Y$ were measured in different units so that $\sigma_Y$ doubled with $\rho$ unchanged, which of $\rho$, $\beta_1$ and $R^2$ would change?
</div>

<div class="exercise" markdown="1">
#### Problem 6

Let $X$ and $Y$ be standardized (mean 0, variance 1) with correlation $\rho=0.7$, and $Y \mid X \sim \text{Normal}(\rho X,\ 1-\rho^2)$.

1. Compute $E[Y\mid X=2]$.
1. A school notices that students who scored in the top 5% on the first exam did worse, on average, on the second, and concludes the exams are demoralizing high achievers. Give the statistical explanation.
1. For which value of $\rho$ would the effect disappear entirely?
</div>

<div class="exercise" markdown="1">
#### Problem 7

For each study, name the design and say whether a difference in average $Y$ between groups can be read causally. If not, name a plausible confounder.

1. Patients are assigned to a drug or a placebo by a coin flip; blood pressure is measured six weeks later.
1. A cohort of adults reports its coffee consumption and is followed for twenty years to see who develops heart disease.
1. A survey finds that adults who own more books have higher incomes.
1. In (c), a researcher proposes to "control for" the number of bookshelves in the home. Explain why this does not fix the problem.
</div>

<div class="exercise" markdown="1">
#### Problem 8

The true model is $Y = 1 + 2X_1 + 4X_2 + \epsilon_1$, and the predictors satisfy $X_2 = 0.5 + cX_1 + \epsilon_2$, with $E[\epsilon_1]=E[\epsilon_2]=0$.

1. A researcher only has $X_1$ and regresses $Y$ on it alone. Give $\beta_1'$ in terms of $c$.
1. Evaluate $\beta_1'$ for $c=0.5$ and for $c=-0.5$.
1. For which values of $c$ does this model exhibit Simpson's paradox?
1. In one sentence, say what the researcher would wrongly conclude when $c=-0.6$.
</div>

<div class="exercise" markdown="1">
#### Problem 9

Monthly rent $Y$ (dollars) is regressed on floor area $X_1$ (square feet) and distance from campus $X_2$ (miles), giving

```text
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        400.2000     55.100      7.264      0.000     291.800     508.600
x1             1.2000      0.140      8.571      0.000       0.925       1.475
x2           -90.0000     41.000     -2.195      0.029    -170.600      -9.400
```

with $R^2=0.62$.

1. State the interpretation of the coefficient on $X_1$, being explicit about what is held fixed.
1. Predict the rent of a 700 sq ft apartment 1.5 miles from campus.
1. Which coefficient is more precisely determined, relative to its size? Justify from the output.
1. A third predictor, "number of rooms," is added and the standard error on $X_1$ triples while $R^2$ barely moves. What is going on, and what does it say about the joint sample distribution of the coefficients?
1. The landlord adds a categorical predictor for neighbourhood, with five neighbourhoods. How many new columns appear in the model, and how would you compute the expected rent difference between two non-baseline neighbourhoods?
</div>


---

## Final

The final is **cumulative**: everything on both midterms, plus [Unit 5]({{ '/unit5/' | relative_url }}) and [Unit 6]({{ '/unit6/' | relative_url }}). It is weighted toward Units 5 and 6, since those have not been examined before, but you should assume anything from Units 1&ndash;4 can reappear.

### What you need to know, beyond the midterms

**Unit 5.**

- **Interactions**: what $J_{1,2}X_1X_2$ does to the slope, how to interpret it, and why centring a predictor makes the other coefficients interpretable again.
- **Residual plots**: why residuals are plotted against the *fitted value* and not against $Y$, what an adequate plot looks like, and what a U-shape or a tilt tells you.
- **Feature maps**: writing a nonlinear $f$ as $\sum_i\beta_i\phi_i(x)$, deciding whether a given $f$ can be fit with linear regression machinery, and building the design matrix.
- **Cross-validation**: training versus test error, why training error only ever decreases, and why test error is U-shaped.
- **Bias-variance**: computing MSE, the decomposition into variance plus squared bias, and saying which way each term moves as a model gains parameters.
- **Orthogonal features**: what orthogonality means, that it depends on the distribution of $X$, and why it makes $\hat\beta_j$ insensitive to which other features are included. Fourier features and the periodogram.

**Unit 6.**

- The conceptual difference between the frequentist and Bayesian treatments of a parameter.
- Prior, likelihood, posterior, evidence &mdash; be able to name each piece of a given calculation.
- The Bernoulli model with a Uniform (or Beta) prior, and the Normal model with a Normal prior and known variance. **Do not memorize the formulas.** Understand the derivations; you may be asked about a single step or how to set one up.
- How the posterior mean interpolates between the prior mean and the data, and what happens in the limits $N\to\infty$, $\tau\to0$ and $\tau\to\infty$.
- **Ridge regression**, the penalty $\lambda\sum_j\beta_j^2$, and the correspondence $\lambda = (\sigma/\tau)^2$ with a Normal prior.
- Computing a regularized estimator by hand in a simple case, such as the sample mean.

There may be extra-credit questions drawn from [Unit 7]({{ '/unit7/' | relative_url }}), depending on how much we cover at the end of term.

### Practice exams

- [Final, 2024]({{ '/public/exam_practice/final_2024C.pdf' | relative_url }}) (ignore the red text in the instructions; do all problems)
- [Practice final]({{ '/public/exam_practice/final_practice.pdf' | relative_url }})
- [Additional practice problems]({{ '/public/exam_practice/final_additional_practice_problems.pdf' | relative_url }})

Also work every **Drill** and end-of-unit **Problem** in Units 5 and 6.
