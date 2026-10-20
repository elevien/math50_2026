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
- Sample spaces, outcomes and events. 
- Read a joint distribution off a table and compute marginal, joint and conditional probabilities from it. Determine if variables are independent. 
- Basic concepts of continuous distributions: The definition of a **density**, **CDF** and how probabilities are calculated. 
- Go from a model written in $\sim$ notation, such as $Y \sim \text{Bernoulli}(q)$ and $X \mid Y \sim \text{Bernoulli}(Y/4 + 1/4)$, to the joint distribution, and back.
- Read and produce the output of simple Python code (arrays, for loops, if statements, `np.random`). Emphasis will be placed on translating simulations to probability statements.

**Unit 2**

- Know propeties of expectation and compute $E[Y]$, $E[g(Y)]$, $\operatorname{var}(Y)$ and the coefficient of variation for a discrete random variable, by hand as well as conditional versions. 
- Use the tower property and the law of total variance to compute $E[Y]$ and $\operatorname{var}(Y)$ from conditional moments.
- Standardize a Normal variable, and use the bell curve shape (68 / 95 / 99.7) to estimate probabilities.
- The distribution of $aX+b$, and of a sum of independent Normals.
- The single-predictor model $Y \mid X \sim \text{Normal}(\beta_0+\beta_1X, \sigma^2)$, its equivalent form $Y = \beta_0+\beta_1X+\epsilon$, and what each parameter controls.
- Why $\beta_1 = E[Y\mid X{=}x+1]-E[Y\mid X{=}x]$.
- Assumptions of linear regression model
- $\operatorname{cov}(X,Y)=\beta_1\sigma_X^2$


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
2. Suppose instead you are told $E[Y]=76.4$ and $E[Y\mid\text{morning}]=72$, but not the afternoon mean. Can you recover it from this information?
3. In one sentence, say why $E[Y]$ is *not* the average of $72$ and $80$.
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

Let $X \sim \text{Bernoulli}(1/2)$ and $Y \mid X \sim \text{Normal}(1+3X,\ 4)$. Three students describe the marginal distribution of $Y$:

- **Student A:** $Y \sim \text{Normal}(2.5,\ 4)$, since the noise variance is $4$.
- **Student B:** $Y \sim \text{Normal}(2.5,\ 6.25)$.
- **Student C:** $Y$ is not Normal, but $E[Y]=2.5$ and $\operatorname{var}(Y)=6.25$.

1. Compute $E[Y]$ using the tower property and $\operatorname{var}(Y)$ using the law of total variance, $\operatorname{var}(Y)=E[\operatorname{var}(Y\mid X)]+\operatorname{var}(E[Y\mid X])$.
1. Which student is correct? For each of the others, say precisely what they get wrong.
1. Now suppose $X \sim \text{Normal}(1/2,\ 1/4)$ instead (the same mean and variance as the Bernoulli), with $Y \mid X$ unchanged. What is the distribution of $Y$ now, and which student is correct in this case? What does this say about which features of $Y$ depend only on the mean and variance of $X$, and which depend on its whole distribution?
</div>

<div class="exercise" markdown="1">
#### Problem 10

An experiment tests whether *believing* an activity is exercise improves health, independent of any change in actual physical activity. Housekeeping staff at a single hotel are split into a control group and a treatment group; the treatment group is told that their daily work already meets recommended exercise guidelines, while the control group is told nothing. Several weeks later, health measurements (e.g. blood pressure) are taken for both groups and the group averages are compared.

1. The description doesn't say how workers were assigned to the two groups. What is the best way to do this assignment, and why?
2. Suppose that instead, workers were free to choose which group they wanted to join. Using the definition of exogeneity, explain why the difference between the two group averages would then be hard to interpret as the effect of the message itself.
3. Even with the assignment from (a), give one reason that having both groups work at the same hotel could still bias the comparison, and suggest a fix.
</div>

<div class="exercise" markdown="1">
#### Problem 11

Consider the following code.

```python
n = 100000
x = np.random.choice([0, 1], size=n)
y = np.random.normal(3 + 4*x, 2, n)

result = np.mean(y[x == 1]) - np.mean(y[x == 0])
print(result)
```


1. Write down the model being simulated for `y` in the form $Y\mid X \sim \text{Normal}(\beta_0+\beta_1X,\ \sigma^2)$, giving the values of $\beta_0$, $\beta_1$ and $\sigma^2$.
1. What is `result` an estimate of, in terms of $\beta_0$ and $\beta_1$?
1. In `numpy`, `np.cov(x,y)[0,1]` returns the covariance between `x` and `y`. Write down numpy code that gives another way of computing `result`. 
</div>

<div class="exercise" markdown="1">
#### Problem 12

Suppose $X \sim \text{Normal}(\mu_X, 4)$ and $Y \mid X \sim \text{Normal}(3X,\ 64)$.

1. Compute $\operatorname{cov}(X,Y)$.
1. Compute $\operatorname{var}(Y)$.
</div>

<div class="exercise" markdown="1">
#### Problem 13

Let $X\sim\text{Bernoulli}(1/2)$ and $Y\mid X\sim\text{Uniform}(X,2)$.

1. Compute $E[Y]$. 
2. Write Python code to generate samples from this model. 
</div>

<div class="exercise" markdown="1">
#### Problem 14

Consider the following code.

```python
n = 100_000
x = np.random.choice([0, 1], size=n, p=[0.3, 0.7])

y = np.empty(n)
y[x == 0] = np.random.uniform(0, 2, len(y[x == 0]))
y[x == 1] = np.random.uniform(1, 4, len(y[x == 1]))

z = np.empty(n)
z[x == 0] = np.random.uniform(-1, 1, len(z[x == 0]))
z[x == 1] = np.random.uniform(0, 6, len(z[x == 1]))

print(np.mean(z))
```

1. Write down the probability model being simulated for $(X,Y,Z)$, in $\sim$ notation. Are $Y$ and $Z$ independent? 
2. What number will `np.mean(z)` print, approximately? 
</div>

<div class="exercise" markdown="1">
#### Problem 15

Consider the following code.

```python
n = 100000
x = np.random.uniform(0, 4, n)
y = np.random.normal(2*x + 1, np.sqrt(3), n)

print(np.var(y))
```

1. Write down the probability model being simulated, in $\sim$ notation.
2. What number will `print(np.var(y))` show, approximately? 
</div>

<div class="exercise" markdown="1">
#### Problem 16

Let $X \sim \text{Bernoulli}(1/3)$ and $Y \mid X \sim \text{Uniform}(0,\,1+X)$.

1. Compute $E[Y\mid X=0]$ and $E[Y\mid X=1]$, and then compute $E[Y]$ using the tower property.
1. Compute $E[XY]$.
1. Compute $\operatorname{cov}(X,Y)$.
</div>

<div class="exercise" markdown="1">
#### Problem 17

The **law of total variance** says that for random variables $X$ and $Y$,

$$ \operatorname{var}(Y) = E\big[\operatorname{var}(Y\mid X)\big] + \operatorname{var}\big(E[Y\mid X]\big). $$

The first term is the average spread of $Y$ within each value of $X$, and the second is the spread of the conditional means across values of $X$.

1. Derive the law of total variance. (Hint: start from $\operatorname{var}(Y)=E[Y^2]-E[Y]^2$)
2. Use it to compute $\operatorname{var}(Y)$ when $X \sim \text{Bernoulli}(1/3)$ and $Y \mid X \sim \text{Uniform}(0,\,1+X)$.
</div>

<div class="exercise" markdown="1">
#### Problem 18

Let $Z_X=(X-\mu_X)/\sigma_X$ and $Z_Y=(Y-\mu_Y)/\sigma_Y$ be the standardized versions of $X$ and $Y$. The **correlation** $\rho$ of $X$ and $Y$ is defined as the regression coefficient of $Z_X$ in the regression with response $Z_Y$ and predictor $Z_X$, that is, $E[Z_Y\mid Z_X]=\rho Z_X$.

Suppose $Y \mid X \sim \text{Normal}(\beta_0+\beta_1X,\ \sigma_\epsilon^2)$.

1. Derive $\rho=\beta_1\sigma_X/\sigma_Y$.
1. Using $\operatorname{cov}(X,Y)=\beta_1\sigma_X^2$, show that $\rho=\operatorname{cov}(X,Y)/(\sigma_X\sigma_Y)$.
1. Show that the noise variance in the regression with response $Z_Y$ and predictor $Z_X$ is $1-\rho^2$, and conclude that $-1\le\rho\le1$.
</div>

<div class="exercise" markdown="1">
#### Problem 19

Answer each of the following in a few sentences.

1. Is it true that $\operatorname{cov}(X,Y)=0$ means $X$ and $Y$ are independent? Justify your answer.
1. We already have the standard deviation to measure spread. What motivates defining the coefficient of variation $\operatorname{CV}=\sigma/\mu$?
1. Why do we define linear regression *models* (a probability distribution for $Y\mid X$) before talking about linear regression on data (fitting a line to points)?
</div>

<div class="exercise" markdown="1">
#### Problem 20

The plot below shows $300$ samples $(x_i,y_i)$ from a linear regression model $Y\mid X\sim\text{Normal}(\beta_0+\beta_1X,\ \sigma_\epsilon^2)$ where $X\in\lbrace0,1\rbrace$. Exactly half of the points have $x=0$. The points are jittered horizontally so that they don't overlap.

<img src="{{ '/public/exam_practice/midterm1_binary_scatter.png' | relative_url }}" alt="Scatter plot of y against a binary x with gridlines" style="max-width: 100%; width: 480px;">

1. Estimate $E[Y\mid X=0]$ and $E[Y\mid X=1]$ from the plot, and use them to estimate $\beta_0$ and $\beta_1$.
2. Estimate $\sigma_\epsilon$ visually from this plot.
3. Using your estimates, compute $\operatorname{cov}(X,Y)$ and $\operatorname{var}(Y)$.
4. Now relabel the groups by defining $X'=1-X$ and fit the same kind of model with $X'$ as the predictor. What are $\beta_0'$, $\beta_1'$, $\operatorname{cov}(X',Y)$ and $\operatorname{var}(Y)$?
</div>

<div class="exercise" markdown="1">
#### Problem 21

Each of the scatter plots below shows $200$ samples $(x_i,y_i)$.

<img src="{{ '/public/exam_practice/midterm1_scatter_models.png' | relative_url }}" alt="Four scatter plots labeled (a) to (d)" style="max-width: 100%; width: 640px;">

1. For each plot, say whether the data could plausibly have been generated by a linear regression model $Y\mid X\sim\text{Normal}(\beta_0+\beta_1X,\ \sigma_\epsilon^2)$. If not, state the assumption that fails. 
2. For a plot that could have come from such a model, estimate $\beta_0$, $\beta_1$ and $\sigma_\epsilon$ from the plot, and say whether $\operatorname{cov}(X,Y)$ is positive or negative.
3. Can exogeneity be checked by looking at a scatter plot of $(x_i,y_i)$ alone? Explain.
</div>


---

## Midterm 2

Covers [Unit 3]({{ '/unit3/' | relative_url }}) and [Unit 4]({{ '/unit4/' | relative_url }}).

Midterm 2 is not cumulative in the sense of having questions focusing on Unit 1 and 2 material, but intermediate steps may involve those topics.  

### What you need to know

**Unit 3**

- Understand the concept of an **estimators** and where the **sample distribution** comes from. Know the difference between an estimator and the true values of the paramater. 
- Understand how we quantify bias in estimators with MSE and how this breaks down into **bias** and **variance**. Know the definiton of **consistency**. Given a simple estimator, you should be able to determine whether it is unbiased and whether it is consistent. You should also be able to construct an estimator that is one but not the other.
- Know the definitions of **standard error** $\text{se}(\hat\mu)=\sigma/\sqrt N$ and  **confidence intervals**. Understand how these are computed both in theory and in practice (from data).
- **Least squares**: You should understand how to obtaiin the formula for the least squares estimatoes of slope and intercept in a linear regression model. The formula for the sample distribution is given on the formula sheet, but you should understand how this derived. 
- You should be familiar with quantities $R^2$, $\rho^2$, $\rho$ and their meaning, as well as how they are related to each other and the other paramaters.
- Understand how regression models are used to model time series data as AR processes, and be able to apply the formula for regression models to this setting. Understand what it means for a process to be **stationary**

**Unit 4**

- Know the difference between **randomized controlled trials** (RCT) and **cohort study**. 
- Be familiar with the multiple predictor regression model and know the interpretation of regression coefficients as well as how $R^2$ and $\rho^2$ are defined. 
- Know and be able to derive the OVB formula, both for one single omitted variable (given on formula sheet) and many omitted variables (NOT given on formula sheet)
- Know how **simpson's paradox** is explained by OVB formula. 
- Understand adding a predictor never decreases $R^2$ (or $\rho^2$).
- Be able to interpret regression model output from `statsmodels` (you don't need to reproduce the code that does the fit) as we did in class. In particular, know what the various numbers mean and be able to related output from different fitted models using OVB formulas. 
- Understand how to include **categorical predictors** using dummy variables, why one category is dropped. 
- Understand how correlations between predictors influences the joint sample distributions of $(\hat\beta_1,\hat\beta_2)$. 


### Practice problems

<div class="exercise" markdown="1">
#### Problem 1

Let $X_1,\dots,X_{100}$ be iid with $E[X_i]=3$ and $\operatorname{var}(X_i)=4$, and let $S=\sum_{i=1}^{100}X_i$.

1. Compute $E[S]$ and $\operatorname{var}(S)$.
1. Write the approximate distribution of $S$, and say what justifies the approximation.
1. Give the approximate distribution of $\overline X$, and its standard deviation.
</div>

<div class="exercise" markdown="1">
#### Problem 2

Let $X_1,\dots,X_N$ be iid $\text{Bernoulli}(q)$ and write $Y=\sum_i X_i$. For each estimator, state whether it is unbiased, whether it is consistent, and give its standard error.

1. $\hat q_a = Y/N$
1. $\hat q_b = (Y+1)/(N+2)$
1. $\hat q_c = X_1$

Then: which of these is unbiased but *not* consistent, and which is consistent but *not* unbiased? Explain in one sentence each why that combination is possible.

Finally, for $N=10$ and $q=0.3$, compute the MSE of $\hat q_a$ and of $\hat q_b$, splitting each into variance and squared bias. Which has the smaller MSE?
</div>

<div class="exercise" markdown="1">
#### Problem 3

A quantity is Normally distributed with known $\sigma=6$. You collect $n=36$ samples and observe $\overline X = 21.5$.

1. Compute $\text{se}(\hat\mu)$ and the approximate 95% confidence interval.
1. How many samples would you need for the 95% confidence interval to be no wider than $\overline X \pm 0.5$?
1. In practice $\sigma$ is unknown. If the sample standard deviation of the $36$ samples is $\hat\sigma=5.1$, recompute the 95% confidence interval.
</div>

<div class="exercise" markdown="1">
#### Problem 4

Fit a single-predictor regression to the four points $(1,1)$, $(2,3)$, $(3,4)$, $(4,6)$.

1. Compute $\hat\beta_1$ and $\hat\beta_0$.
1. Compute the four residuals and the sum of squared residuals (SSR).
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

The data is generated from $Y = 1 + 2X_1 + 4X_2 + \epsilon_1$, where the predictors satisfy $X_2 = 0.5 + cX_1 + \epsilon_2$, with $E[\epsilon_1]=E[\epsilon_2]=0$.

1. A researcher only has $X_1$ and fits a regression with response $Y$ and $X_1$ as the only predictor. Give $\beta_1'$ in terms of $c$.
1. Evaluate $\beta_1'$ for $c=0.5$ and for $c=-0.5$.
1. For which values of $c$ does this model exhibit Simpson's paradox?
1. In one sentence, say what the researcher would wrongly conclude when $c=-0.6$.
</div>

<div class="exercise" markdown="1">
#### Problem 9

A regression with monthly rent $Y$ (dollars) as the response and floor area $X_1$ (square feet) and distance from campus $X_2$ (miles) as the predictors gives

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

<div class="exercise" markdown="1">
#### Problem 10

Consider the AR(1) process $Y_t=\rho Y_{t-1}+\epsilon_t$ with $\epsilon_t\sim\text{Normal}(0,\sigma_\epsilon^2)$.

1. What does it mean for the process to reach a stationary distribution as $t\to\infty$, and for which values of $\rho$ does this happen?
1. Assuming stationarity, show that $\operatorname{var}(Y_t)=\sigma_\epsilon^2/(1-\rho^2)$.
1. In the regression with response $Y_t$ and predictor $Y_{t-1}$, show that the slope is $\rho$ and that $R^2=\rho^2$.
1. Use the omitted variable bias formula to find the slope when the response is $Y_t$ and the only predictor is $Y_{t-2}$. What is the slope when the only predictor is $Y_{t-k}$?
1. With $\rho=0.9$, $\sigma_\epsilon=2$ and $N=200$ observations, compute the approximate standard error of the slope in part 3.
</div>

<div class="exercise" markdown="1">
#### Problem 11

The data is generated from $Y=2+X_1+3X_2-2X_3+\epsilon$, where $\epsilon$ is uncorrelated with the predictors. In a single-predictor regression with predictor $X_1$, the slope is $0.5$ when the response is $X_2$ and $1.2$ when the response is $X_3$.

1. Derive the formula for the slope $\beta_1'$ of the regression with response $Y$ and $X_1$ as the only predictor, when both $X_2$ and $X_3$ are left out.
1. Compute $\beta_1'$. Does $X_1$ look more or less important than it is?
1. For what value of the slope with response $X_3$ and predictor $X_1$ would the regression with response $Y$ and only predictor $X_1$ show no association at all?
</div>

<div class="exercise" markdown="1">
#### Problem 12

Treat $x_1,\dots,x_N$ as fixed and let $Y_i=\beta_0+\beta_1x_i+\epsilon_i$ with independent $\epsilon_i\sim\text{Normal}(0,\sigma_\epsilon^2)$.

1. Show that $E[\hat\beta_1]=\beta_1$.
1. Show that $\operatorname{var}(\hat\beta_1)=\sigma_\epsilon^2/\sum_i(x_i-\bar x)^2$, and explain why this is approximately $\sigma_\epsilon^2/(N\sigma_X^2)$, as on the formula sheet.
1. You can choose $N$ values of $x$ anywhere in $[0,1]$. Which choice gives the smallest standard error for $\hat\beta_1$?
</div>

<div class="exercise" markdown="1">
#### Problem 13

A single-predictor regression with response $Y$ and predictor $X$ has slope $\beta_1=-1.4$, where $\sigma_X=5$ and $\sigma_Y=10$.

1. What is the correlation $\rho$ between $X$ and $Y$?
1. What is $\rho^2$, and what does it tell you?
1. If we swap the roles, with $X$ as the response and $Y$ as the predictor, what are the slope and $\rho^2$?
1. What is the standard deviation of the noise, $\sigma_\epsilon$?
</div>

<div class="exercise" markdown="1">
#### Problem 14

1. Explain why adding a predictor to a regression can never increase the sum of squared residuals (SSR), and so can never decrease $R^2$. (Hint: what happens if the new coefficient is set to zero?)
1. A student adds a column of pure random noise, unrelated to $Y$, as a predictor. What happens to $R^2$ in the data, and to the population quantity $\rho^2$?
1. With response $Y$, the regression with only predictor $X_1$ has $R^2=0.30$, and the regression with only predictor $X_2$ has $R^2=0.25$. What can you say about $R^2$ for the regression with both predictors? What is it if $X_1$ and $X_2$ are uncorrelated?
</div>

<div class="exercise" markdown="1">
#### Problem 15

Let $X_1,\dots,X_N$ be iid $\text{Normal}(\mu,\sigma^2)$. Consider the estimators
$$\hat\mu_1=\overline X+\frac1N,\qquad \hat\mu_2=\frac{X_1+X_N}{2},\qquad \hat\mu_3=\frac{N}{N+1}\overline X.$$

1. For each, compute the bias, the variance and the MSE.
1. Which are consistent? Which are unbiased?
1. For which values of $\mu$ does $\hat\mu_3$ have a smaller MSE than $\overline X$? Evaluate your answer for $N=10$.
</div>

<div class="exercise" markdown="1">
#### Problem 16

In a poll of $N=400$ voters, $216$ support a ballot measure.

1. Estimate the proportion $q$ who support the measure, and compute its standard error from the data.
1. Compute an approximate 95% confidence interval for $q$.
1. How many voters would need to be polled for the 95% confidence interval to be $\hat q\pm0.02$? (Use your estimate of $q$.)
</div>

<div class="exercise" markdown="1">
#### Problem 17

A school offers an optional tutoring program. Students who chose to attend have final grades $8$ points higher, on average, than students who did not.

1. Is this a randomized controlled trial or a cohort study? Name a plausible confounder.
1. A regression with final grade as the response and attendance ($0/1$) and prior GPA as the predictors gives a coefficient of $3$ on attendance. Students who attended had prior GPAs $0.5$ higher, on average. Use the omitted variable bias formula to find the coefficient on prior GPA.
1. Describe a study design in which the single-predictor comparison would estimate the effect of tutoring, and explain using the omitted variable bias formula why it works.
</div>

<div class="exercise" markdown="1">
#### Problem 18

A regression with daily bike rentals $Y$ (hundreds) as the response and temperature (°C) and season as the predictors, with winter as the baseline, gives:

```text
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         10.0000      1.200      8.333      0.000       7.648      12.352
temp           1.5000      0.080     18.750      0.000       1.343       1.657
spring         4.0000      1.100      3.636      0.000       1.844       6.156
summer         6.5000      1.250      5.200      0.000       4.050       8.950
fall           5.0000      1.150      4.348      0.000       2.746       7.254
```

1. Why are there three season variables and not four?
1. Predict the rentals on a summer day at $20$°C.
1. What is the expected difference between a summer day and a fall day at the same temperature?
1. If summer were used as the baseline instead, what would the constant and the coefficients on winter, spring, fall and temperature be?
</div>

<div class="exercise" markdown="1">
#### Problem 19

Two standardized predictors $X_1,X_2$ have correlation $\rho$, and $Y=\beta_1X_1+\beta_2X_2+\epsilon$ with $\sigma_\epsilon=2$ and $N=100$. For this model $\text{se}(\hat\beta_1)=\sigma_\epsilon/\sqrt{N(1-\rho^2)}$ and $\text{corr}(\hat\beta_1,\hat\beta_2)=-\rho$.

1. Compute $\text{se}(\hat\beta_1)$ for $\rho=0$, $0.8$ and $0.99$.
1. Sketch the sample distribution of $(\hat\beta_1,\hat\beta_2)$ for $\rho=0$ and for $\rho=0.8$.
1. For $\rho=0.99$, a colleague finds that neither coefficient is significant and concludes that neither predictor is related to $Y$. Explain why this can be wrong, and which combination of the coefficients is still estimated precisely.
1. Which combination is estimated precisely when $\rho=-0.8$?
</div>

<div class="exercise" markdown="1">
#### Problem 20

A random walk is $Y_t=Y_{t-1}+\epsilon_t$ with $Y_0=0$ and independent $\epsilon_t\sim\text{Normal}(0,\sigma_\epsilon^2)$.

1. Compute $\operatorname{var}(Y_t)$ as a function of $t$. What happens to it as $t\to\infty$? Does the random walk have a stationary distribution?
1. For comparison, take the AR(1) process $Y_t=\rho Y_{t-1}+\epsilon_t$ with $\lvert\rho\rvert<1$ and $Y_0=0$. Compute $\operatorname{var}(Y_t)$ and its limit as $t\to\infty$.
1. For the log price of bitcoin, the slope of the regression with response $Y_t$ and predictor $Y_{t-1}$ is about $0.9996$. What does this suggest about the process?
1. Let $D_t=Y_t-Y_{t-1}$ be the daily changes of a random walk. What is the distribution of $D_t$, and does it change with $t$? What is the slope of the regression with response $D_t$ and predictor $D_{t-1}$?
</div>

<div class="exercise" markdown="1">
#### Problem 21

The diagram shows a regression with response $Y$ and predictors $X_1$, $X_2$ and $X_3$. The arrows into $Y$ are the coefficients in the model with all three predictors. The arrows from $X_1$ to $X_2$ and $X_3$ are the slopes of single-predictor regressions with $X_1$ as the predictor and $X_2$ or $X_3$ as the response.

<img src="{{ '/public/exam_practice/midterm2_ovb_1.svg' | relative_url }}" alt="Diagram with predictors X1, X2, X3 stacked vertically and response Y. Arrows into Y: from X1 labeled 2, from X2 labeled 3, from X3 labeled -1. Arrows from X1 to X2 labeled 0.5 and from X1 to X3 labeled 2." style="width: 100%; max-width: 480px;">

1. Compute the slope $\beta_1'$ of the regression with response $Y$ and $X_1$ as the only predictor.
1. Which left-out predictor changes the coefficient on $X_1$ the most?
1. If the coefficient on $X_3$ were $-3$ instead of $-1$, what would $\beta_1'$ be? What is this an example of?
1. For what value of the slope from $X_1$ to $X_3$ would $\beta_1'=0$?
</div>

<div class="exercise" markdown="1">
#### Problem 22

The diagram shows a model for exam scores, with arrows as in Problem 21.

<img src="{{ '/public/exam_practice/midterm2_ovb_2.svg' | relative_url }}" alt="Diagram with predictors hours studied (X1), prior GPA (X2) and hours of sleep (X3) stacked vertically and response exam score (Y). Arrows into exam score: from hours studied labeled 2, from prior GPA labeled 10, from hours of sleep labeled 1. Arrows from hours studied to prior GPA labeled 0.1 and to hours of sleep labeled -0.5." style="width: 100%; max-width: 560px;">

1. A study records only hours studied and exam scores. What slope will it find for hours studied?
1. Students who study more tend to have higher prior GPAs but sleep less. Explain how each of these changes the coefficient on hours studied.
1. Does the study overstate or understate the effect of an extra hour of studying, holding prior GPA and sleep fixed?
</div>

<div class="exercise" markdown="1">
#### Problem 23

The diagram shows a regression with four predictors, with arrows as in Problem 21.

<img src="{{ '/public/exam_practice/midterm2_ovb_3.svg' | relative_url }}" alt="Diagram with predictors X1, X2, X3, X4 stacked vertically and response Y. Arrows into Y: from X1 labeled 1, from X2 labeled 4, from X3 labeled -2, from X4 labeled 5. Arrows from X1 to X2 labeled 0.5, to X3 labeled 1, and to X4 labeled 0." style="width: 100%; max-width: 480px;">

1. Compute the slope $\beta_1'$ of the regression with response $Y$ and $X_1$ as the only predictor.
1. Your answer should equal the coefficient on $X_1$ in the full model, even though $X_2$, $X_3$ and $X_4$ all affect $Y$. Explain why $X_4$ contributes nothing, and what happens with $X_2$ and $X_3$.
1. What would $\beta_1'$ be if the slope from $X_1$ to $X_4$ were $0.2$ instead of $0$?
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
