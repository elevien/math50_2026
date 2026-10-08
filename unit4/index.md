---
layout: notes
title: "Unit 4: Regression with Multiple Predictors"
unit_title: "Unit 4"
unit_subtitle: "Study design, multiple predictors, covariance matrices, collinearity, and categorical data"
toc:
  - {href: "#sec-4-1", label: "4.1 Study design & confounding"}
  - {href: "#sec-4-2", label: "4.2 Multiple-predictor regression"}
  - {href: "#sec-4-3", label: "4.3 Covariance matrix & estimation"}
  - {href: "#sec-4-4", label: "4.4 Sample distribution & collinearity"}
  - {href: "#sec-4-5", label: "4.5 Categorical predictors"}
  - {href: "#problems", label: "Problems"}
---

<div class="unit-overview" markdown="1">

This unit covers linear regression with **multiple predictors**. We will first discuss the idea of a **confounding variable** and study design, which motivates models with multiple predictors. Many new ideas emerge when we add more predictors. Most importantly, regression coefficients now need to be interpreted as average differences with all *other* predictors held fixed, so the relationship between predictors plays a big role. After this unit you should understand how adding and removing predictors changes the others. We'll talk about the joint sample distribution of regression coefficients and **collinearity**, understanding these ideas from multiple angles (linear algebra, geometry, and simulation). We'll also introduce **categorical predictors**, which let us handle qualitative data. We will also discuss the linear algebra view of fitting a regression model. 

#### Concepts

Study design, randomized controlled trials and observational studies, confounding, regression with multiple predictors, controlling for a variable, the covariance matrix and the OLS estimator, Simpson's paradox, the effect of adding predictors on $R^2$, the joint sample distribution of the coefficients, collinearity, categorical predictors and dummy variables.

#### Things to practice

- Derive the linear system relating the regression coefficients to the covariances (generalizing the single-predictor case).
- Interpret the output of a fitted model with multiple predictors, including qualitative predictors.
- Predict how adding a predictor changes the regression coefficient of an existing predictor, given some knowledge of the predictors.
- Identify collinearity and understand its effect on the sample distribution.
- Include categorical predictors in a regression model.

</div>

<p class="pdf-link"><a href="unit4.pdf">Unit 4 notes pdf</a> <a href="https://colab.research.google.com/github/elevien/math50_2026/blob/main/unit4/unit4.ipynb">Unit 4 notebook</a></p>

Now we're ready to study <span class="term">[linear regression with multiple predictors](https://en.wikipedia.org/wiki/Multiple_linear_regression)</span>. Much of the theory carries over from the single-predictor case, but there are new subtleties in how we *interpret* the predictors. In particular, a regression coefficient represents an average difference in the response variable when all other predictors are held fixed: this is the idea of **controlling for** another variable. We'll also work out what happens to the model when we add or remove predictors.

## 4.1 Study design and confounding {#sec-4-1}

A central question in many scientific contexts is: Does changing variable $X$ cause a change in variable $Y$? 

It should be clear that two variables can be strongly related in a dataset without either one causing the other. The classic textbook example is the relationship between ice cream sales and drowning. They are associated, but tempurature is a  <span class="term">[confounding variable](https://en.wikipedia.org/wiki/Confounding)</span> (or confounder).  Whether we are able to interpret a relationship as causal depends less on the regression formula itself than on how the data were collected, or what we call the <span class="term">[study design](https://en.wikipedia.org/wiki/Research_design)</span>. 

Here, to give a taste of this topic, we will discuss only two types of study designs:

<span class="term">[Randomized controlled trial](https://en.wikipedia.org/wiki/Randomized_controlled_trial) (RCT).</span> If we randomly assign each unit's value of $X$ &mdash; e.g. assigning patients to treatment or control &mdash; then $X$ is, by construction, independent of other variables that might affect $Y$. Any systematic difference in $Y$ between the groups can then be attributed to $X$ itself.

<span class="term">[Cohort study](https://en.wikipedia.org/wiki/Cohort_study) (prospective [observational](https://en.wikipedia.org/wiki/Observational_study)).</span> Often randomization is not possible or ethical. A cohort study follows a group forward in time and observes who ends up with which value of $X$ and what $Y$ they go on to have. Because $X$ was observed rather than assigned, it may be related to other causes of $Y$.

In a cohort study we cannot know whether there is another variable confounder $X_2$, correlated with $X$, which also has an effect on $Y$.  It can produce an association between $X$ and $Y$ even if $X$ has no causal effect on $Y$, or it can inflate, shrink, or reverse a real effect. In an RCT, randomization makes $X$ independent of *every* such $X_2$, measured or not. (We call it $X_2$ rather than a new letter because the fix will be to include it as a second predictor in the regression.)

<div class="example" markdown="1">
#### Example

Recall the population of people from Unit 1, who differ in eye color and whether they smile. Let $X=1$ if a person has blue eyes and $Y=1$ if they smile:

<figure style="margin: 1em 0;">
<img src="{{ '/unit4/fig/people_by_eye.svg' | relative_url }}" alt="Twenty-four cartoon people in two groups of twelve, blue-eyed and brown-eyed. Seven of the twelve blue-eyed people smile; five of the twelve brown-eyed people smile." style="width: 100%; max-width: 640px;">
</figure>

Blue-eyed people smile more often: $P(Y=1\mid X=1)=7/12$ but $P(Y=1\mid X=0)=5/12$. Do blue eyes make people smile? Now also record $X_2=1$ if a person has ears and $X_2=0$ if they do not:

<figure style="margin: 1em 0;">
<img src="{{ '/unit4/fig/people_by_ears.svg' | relative_url }}" alt="The same people sorted by ears and eye color. Among people with ears, 6 of 8 blue-eyed and 3 of 4 brown-eyed people smile. Among people without ears, 1 of 4 blue-eyed and 2 of 8 brown-eyed people smile." style="width: 100%; max-width: 640px;">
</figure>

Among people with the *same* ears, eye color makes no difference: $3/4$ smile with ears and $1/4$ without, whatever their eyes. Blue-eyed people are simply more likely to have ears ($8/12$ vs. $4/12$), and ears go with smiling. So $X_2$ is a confounder: it produces an association between eye color and smiling even though, among people with the same ears, there is none. In regression form (see the next section), the conditional averages are exactly

$$ E[Y\mid X]=\frac5{12}+\frac16X, \qquad E[Y\mid X,X_2]=\frac14+0\cdot X+\frac12X_2, $$

so once we control for ears, the coefficient of eye color drops from $1/6$ to $0$.
</div>

<details class="optional-section" open markdown="1">
<summary><strong>Aside: other study designs</strong> <span class="optional-badge">(optional)</span></summary>

There are many other study designs: A <span class="term">[case-control study](https://en.wikipedia.org/wiki/Case%E2%80%93control_study)</span> works backward from the outcome by sampling people who already have $Y=1$ (cases) and $Y=0$ (controls), then looks back at their history of $X$. A <span class="term">[cross-sectional study](https://en.wikipedia.org/wiki/Cross-sectional_study)</span> measures $X$ and $Y$ at a single point in time, with no notion of before and after. A <span class="term">[natural experiment](https://en.wikipedia.org/wiki/Natural_experiment)</span> looks for a real-world situation where $X$ was effectively randomized by circumstance, such as a lottery, policy threshold, or eligibility rule.

</details>

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Identifying study designs

For each study, name the design and say whether a difference in average $Y$ between the $X$ groups can be read causally.

<ol type="a">
  <li>Patients are assigned to a drug or a placebo by a coin flip, and blood pressure is measured six weeks later.</li>
  <li>A group of adults is recruited, their coffee drinking is recorded, and they are followed for twenty years to see who develops heart disease.</li>
  <li>Everyone in a town is surveyed on one day about both their exercise habits and their current weight.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Confounding variables

A dataset shows that towns with more ice-cream shops per capita have more drownings per capita.

<ol type="a">
  <li>Propose a variable $X_2$ that plausibly affects both, and explain why it fits the definition of a confounder.</li>
  <li>Would randomly assigning the number of ice-cream shops remove the problem? Explain in one sentence.</li>
  <li>Give one variable that is affected by $X$ and $Y$ but is <em>not</em> a confounder.</li>
</ol>
</div>

</details>

## 4.2 Multiple-predictor regression {#sec-4-2}

The real power of regression comes from models of the form

$$ Y = \beta_0 + \sum_{i=1}^K \beta_iX_i + \epsilon, \qquad \epsilon \sim \text{Normal}(0,\sigma^2), $$

where $X_i$ is one of $K$ predictor variables. Equivalently,

$$ Y \mid (X_1=x_1,\dots,X_K=x_K) \sim \text{Normal}\Big(\beta_0 + \sum_{i=1}^K \beta_iX_i,\ \sigma^2\Big). $$

You might see the shorthand $Y \sim \text{LR}(X,\beta,\sigma^2)$.

In these notes, our goal is to answer:

1. What are the estimators of the parameters in this model?
2. How do we interpret the <span class="term">[regression coefficients](https://en.wikipedia.org/wiki/Linear_regression)</span> $\beta_i$?
3. What is the sample distribution of the regression coefficients?
4. How do correlations between predictors influence the answers to these questions?

<div class="example" markdown="1">
#### Example (two-predictor regression in `statsmodels`)

<u>Question:</u> generate data from the two-predictor model

$$ Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \epsilon, \qquad \epsilon \sim \text{Normal}(0,\sigma^2), $$

and fit the linear regression $Y \sim X_1+X_2$ using `statsmodels`.

<u>Solution:</u>

```python
import numpy as np
import statsmodels.api as sm

# --- simulate data ---
n = 300
rng = np.random.default_rng(0)

beta0, beta1, beta2, sigma = 1.0, 2.0, -1.0, 0.5
X1 = rng.normal(size=n)
X2 = rng.normal(size=n)
eps = rng.normal(scale=sigma, size=n)
Y = beta0 + beta1*X1 + beta2*X2 + eps

# --- fit OLS: Y ~ X1 + X2 ---
X = sm.add_constant(np.column_stack([X1, X2]))
model = sm.OLS(Y, X)
results = model.fit()
```
</div>

<!--
The output of a multiple-predictor regression looks basically the same as the single-predictor case, except now there's a row for each coefficient. The interpretation of the $p$-value and confidence interval for each coefficient is nearly the same as before &mdash; though we should remember that the $p$-value is testing whether *that particular* predictor's coefficient is zero. The $F$-statistic tests the hypothesis that *all* the predictors are zero; we won't go into more detail since hypothesis testing isn't a big focus of this course.
-->

The interpretation of $R^2$ is the same as before, except now we're comparing the variance conditioned on *all* predictors to the overall variance in $Y$:

$$ R^2 = 1 - \frac{\sum_i r_i^2}{\sum_i (Y_i-\overline Y)^2} \approx 1 - \frac{\text{var}(Y\mid X_1,X_2)}{\text{var}(Y)}, $$

where, in the multi-predictor case,

$$ r_i = Y_i - \Big(\hat\beta_0 + \sum_{k=1}^K \hat\beta_kX_{i,k}\Big). $$

### Basic interpretation and estimation of the parameters

To interpret the parameters, it's easiest to work with just two predictors, as in the example above. The conditional expectation of $Y$ is

$$ E[Y\mid X] = \beta_0 + \beta_1X_1 + \beta_2X_2, $$

where I'm using the shorthand $E[Y\mid X] = E[Y\mid(X_1,X_2)]$ to mean the expectation conditioned on **both** predictors. This is the equation for a flat surface (a plane) over the $(x_1,x_2)$-plane. If we slice the surface in the $x_1$ direction and look at it from the side, we see a line with slope $\beta_1$ (and similarly for $x_2$). This gives the key interpretation:

<div style="text-align:center; margin:1em 0;">$\beta_1$ is the slope of $E[Y\mid X]$ vs. $X_1$ for fixed $X_2$.</div>

The demo below slices the surface both ways: the orange line holds $X_2$ fixed and varies $X_1$, the blue line holds $X_1$ fixed and varies $X_2$. Drag either slider and watch its own line's *slope* stay put while only its height moves &mdash; that invariance is exactly what the statement above says.

Notice that even though we're conditioning on both variables, the slope $\beta_1$ doesn't depend on which value of $X_2$ we condition on &mdash; we get the interpretation of $\beta_2$ by swapping the roles of $X_1,X_2$. This is a consequence of linearity, and it's one of the core assumptions of multiple-predictor regression that we didn't encounter with a single predictor: the "effect" of $X_1$ doesn't depend on the value of $X_2$ (and vice versa).

{% include_relative demos/plane.html %}

<div class="example" markdown="1">
#### Example (children's test scores)

We'll now work with a new example: children's test scores. Imagine we're studying what factors determine children's success in school, in order to design interventions that help struggling students. The predictors are mother's IQ and whether the mother attended high school. Here, the model assumption says that the association between the mother's high school education and test scores isn't influenced by the mother's IQ: if we compare two random children whose mothers have the same IQ but differ in whether they attended high school, the average *difference* in test scores won't depend on the IQ of their mothers &mdash; although the average test score itself will.

<u>Question:</u> fit the data to a linear regression model with two predictors and answer:

<ol type="a">
<li>What are the regression coefficients, and their interpretations?</li>
<li>Based on this analysis, which factor &mdash; IQ or high school education &mdash; seems more predictive of test scores?</li>
<li>Overall, how well do high school education and IQ together predict test scores?</li>
<li>What's the chance a student whose mother has an IQ of $90$ and did not attend high school does better than a student whose mother has an IQ of $110$ and did attend high school?</li>
</ol>

<u>Solution:</u> we get the following output from `statsmodels` (see the companion Colab notebook):

```text
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.214
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         25.7315      5.875      4.380      0.000      14.184      37.279
mom_hs         5.9501      2.212      2.690      0.007       1.603      10.297
mom_iq         0.5639      0.061      9.309      0.000       0.445       0.683
```

<ol type="a">
<li>For the coefficients:
<ul>
<li>$\beta_{\text{hs}} \approx 5.95$: among students whose mothers <strong>have the same IQ</strong>, a student whose mother attended high school scores $5.95$ points higher on average than one whose mother didn't.</li>
<li>$\beta_{\text{iq}} \approx 0.56$: among students whose mothers <strong>have the same high-school education status</strong>, a one-point difference in mother's IQ is associated with a $0.56$-point difference in score, on average.</li>
<li>$\hat\beta_0 \approx 26$: mathematically, the average score of students whose mother didn't attend high school and has zero IQ &mdash; not a meaningful quantity, since nobody has zero IQ, so we can ignore it when interpreting the output.</li>
</ul>
</li>
<li>$\beta_{\text{hs}}$ is smaller, but the two coefficients have different units: $X_{\text{iq}}$ ranges roughly from $70$ to $130$, while $X_{\text{hs}}$ is $0$ or $1$. A unit-free way to compare them is to scale each coefficient by its standard error, $\hat T=\hat\beta/\text{se}(\hat\beta)$: $\hat T_{\text{hs}}\approx5.95/2.21\approx2.7$ vs. $\hat T_{\text{iq}}\approx0.56/0.061\approx9.3$, so the IQ coefficient is far more clearly different from $0$ relative to its uncertainty. Another option is the effect of a one-standard-deviation change in each predictor: $\hat\beta_{\text{hs}}\hat\sigma_{\text{hs}} \approx 2.44$ vs. $\hat\beta_{\text{iq}}\hat\sigma_{\text{iq}} \approx 8.44$. Either way, the association with IQ is stronger (though the comparison isn't perfect, since $X_{\text{hs}}$ is binary).

<figure style="margin: 0.8em 0;">
<img src="{{ '/unit4/fig/kidiq_coefs.svg' | relative_url }}" alt="Left: the two coefficients with 95% intervals on their raw scale; the high-school coefficient (5.95) looks much larger than the IQ coefficient (0.56). Right: the same coefficients in units of their standard errors; the IQ coefficient (9.31) is much further from zero than the high-school coefficient (2.69)." style="width: 100%;">
</figure></li>
<li>The $R^2$ value is $0.214$, so about $21\%$ of the variation in test scores is explained by mother's high-school education and IQ together.</li>
<li>In the Colab notebook, we calculate this probability to be about $25\%$.</li>
</ol>
</div>

This interpretation generalizes to many predictors. In terms of conditional expectations,

$$ \beta_i = E[Y\mid X_1,\dots,X_{i-1},X_i{=}x_i{+}1,X_{i+1},\dots,X_K] - E[Y\mid X_1,\dots,X_{i-1},X_i{=}x_i,X_{i+1},\dots,X_K], $$

a natural extension of the two-predictor formula below.

### Relationship between single- and two-predictor regression coefficients

We can write the two-predictor coefficient explicitly in terms of conditional averages,

$$ \beta_1 = E[Y\mid X_1{=}(x{+}1),X_2] - E[Y\mid X_1{=}x,X_2]. $$

How is this related to covariance? A first guess: just as in the single-predictor case, $\beta_1 = \text{cov}(Y,X_1)/\sigma_{X_1}^2$. After all, slicing the 2D planar surface along $x_1$ gives the same slope $\beta_1$ for every $x_2$, so it stands to reason that regressing on the $(x_1,y)$ points alone should also give slope $\beta_1$. **But this argument silently assumes that when $x_1$ changes, $x_2$ doesn't change with it.** Let's see why that matters with an example.

<div class="example" markdown="1">
#### Example (test scores: multiple vs. single predictor)

We again consider children's test scores, comparing the two-predictor fit above to a fit using only high-school education as a predictor.

<u>Question:</u> what's the difference between the coefficient of $X_{\text{hs}}$ when it's the only predictor and when $X_{\text{iq}}$ is also included? How are the two related?

<u>Solution:</u> using only mother's high-school education, we get $\hat\beta_{\text{hs}}^{\prime} \approx 12$ and $\hat\beta_0^{\prime} \approx 78$ (using $\beta^{\prime}$ for the single-predictor coefficients), so $\hat y = 12X_{\text{hs}} + 78$ &mdash; while with both predictors, the coefficient of $X_{\text{hs}}$ is about half that.

In the one-predictor model, a coefficient of $12$ means a student whose mother went to high school does $12$ points better on average, i.e. we're predicting

$$ E[Y\mid X_{\text{hs}}{=}1] - E[Y\mid X_{\text{hs}}{=}0] = \beta_{\text{hs}}' \approx 12. $$

Compare this to the two-predictor model. The average test score for students whose mothers went to high school is

$$
\begin{aligned}
\hat y_{\text{hs}}
&\approx E[Y\mid X_{\text{hs}}{=}1] \\
&= E[\beta_0+\beta_{\text{hs}}+\beta_{\text{iq}}X_{\text{iq}} \mid X_{\text{hs}}{=}1] \\
&= \beta_0+\beta_{\text{hs}}+\beta_{\text{iq}}E[X_{\text{iq}}\mid X_{\text{hs}}{=}1] \\
&\approx 26+6+0.6\,\overline X_{\text{iq}\mid\text{hs}}.
\end{aligned}
$$

where $\overline X_{\text{iq}\mid\text{hs}}$ is the sample average IQ among mothers who attended high school. Similarly, $\hat y_{\text{no-hs}} = 26+0.6\,\overline X_{\text{iq}\mid\text{no-hs}}$. So, according to the two-predictor model, the average difference between the groups is

$$ \Delta\hat y_{\text{hs}} = 6 + 0.6\big(\overline X_{\text{iq}\mid\text{hs}}-\overline X_{\text{iq}\mid\text{no-hs}}\big), $$

or, in probabilistic notation,

$$ E[Y\mid X_{\text{hs}}{=}1]-E[Y\mid X_{\text{hs}}{=}0] = \beta_{\text{hs}} + \beta_{\text{iq}}\big(E[X_{\text{iq}}\mid X_{\text{hs}}{=}1]-E[X_{\text{iq}}\mid X_{\text{hs}}{=}0]\big). $$

We can compute $\overline X_{\text{iq}\mid\text{hs}}-\overline X_{\text{iq}\mid\text{no-hs}} \approx 10.3$, giving $\Delta\hat y_{\text{hs}} \approx 12$ &mdash; recovering the single-predictor coefficient from the two-predictor model.
</div>

The demo below separates the two slopes, using a binary $X_1$ and continuous $X_2$ (as in the KidIQ example above). The left panel plots $Y$ against $X_2$, split by $X_1$: the two solid lines are the within-group slopes, which should both sit close to $\beta_2$. The right panel plots $Y$ against $X_1$, split by tercile of $X_2$: the solid segments are the within-tercile differences, which should sit close to $\beta_1$. In both panels, the dashed line is the marginal slope from regressing $Y$ on that one predictor alone &mdash; increase the confounding strength $b$ to pull the solid and dashed lines apart.

{% include_relative demos/correlated_predictors.html %}

The important thing above is that the two predictors are *not* independent. If they were, $\overline X_{\text{iq}\mid\text{hs}}-\overline X_{\text{iq}\mid\text{no-hs}}$ would be zero, and the coefficient of $X_{\text{hs}}$ would have to be the same with or without $X_{\text{iq}}$ in the model. This generalizes: for any model where $X_1$ is a binary predictor,

$$ \beta_1' = \beta_1 + \beta_2\big(E[X_2\mid X_1{=}1]-E[X_2\mid X_1{=}0]\big), $$

where $\beta_1^{\prime}$ is the regression coefficient of $X_1$ *without* $X_2$ in the model.

### Simpson's paradox

In the example above, $\beta_1^{\prime}$ and $\beta_1$ can even have different signs, depending on the relationship between the predictors. This effect is called <span class="term">[Simpson's "paradox"](https://en.wikipedia.org/wiki/Simpson%27s_paradox)</span> &mdash; not really a paradox, just a consequence of correlation between predictors shifting the single-predictor coefficient, as illustrated above.

<div class="example" markdown="1">
#### Example (Simpson's paradox with two binary predictors)

Consider two binary predictors $X_1,X_2\in\lbrace0,1\rbrace$ with joint distribution

<table class="prob-table">
<tr><th>$P(X_1,X_2)$</th><th>$X_2=0$</th><th>$X_2=1$</th></tr>
<tr><th>$X_1=0$</th><td>0.4</td><td>0.1</td></tr>
<tr><th>$X_1=1$</th><td>0.1</td><td>0.4</td></tr>
</table>

and suppose $Y\mid(X_1,X_2) = X_1 - 2X_2+\epsilon$.

<u>Question:</u> compute the single-predictor regression coefficient $\beta_1^{\prime}$ of $X_1$ on $Y$.

<u>Solution:</u> first, the conditional means of $X_2$ given $X_1$:

$$ P(X_2{=}1\mid X_1{=}1) = \frac{0.4}{0.1+0.4}=0.8, \qquad P(X_2{=}1\mid X_1{=}0) = \frac{0.1}{0.4+0.1}=0.2, $$

$$ \beta_{1,2} \equiv E[X_2\mid X_1{=}1]-E[X_2\mid X_1{=}0] = 0.8-0.2=0.6. $$

With $\beta_1=1,\beta_2=-2$, the single-predictor slope is

$$ \beta_1' = \beta_1+\beta_2\beta_{1,2} = 1+(-2)(0.6) = -0.2. $$

Note the sign flip: the two-predictor model says increasing $X_1$ *increases* $Y$, but the single-predictor model says the opposite!
</div>

### Effect of adding predictors on $R^2$

Adding a new predictor to a regression model can never *decrease* $R^2$. Let $R^2\_{1\,\mathrm{pred}}$ be the $R^2$ for $Y=\beta_1X_1+\epsilon^{\prime}$, and $R^2\_{2\,\mathrm{pred}}$ the $R^2$ for $Y=\beta_1X_1+\beta_2X_2+\epsilon$. Since $R^2=1-\operatorname{var}(\mathrm{residual})/\operatorname{var}(Y)$, $R^2$ increases whenever the residual variance decreases.

The key point: the error term $\epsilon^{\prime}$ in the reduced (single-predictor) model is *not* the same as $\beta_2X_2+\epsilon$ &mdash; it absorbs the variation in $Y$ that's correlated with $X_2$ but unaccounted for once we drop it. Since $X_1,X_2$ are generally correlated, part of the systematic variation explained by $X_2$ gets treated as noise in the single-predictor model:

$$ \operatorname{var}(\epsilon') = \operatorname{var}(Y\mid X_1) = \operatorname{var}(\beta_1X_1+\beta_2X_2+\epsilon\mid X_1) = \beta_2^2\,\operatorname{var}(X_2\mid X_1)+\sigma_\epsilon^2 \ge \sigma_\epsilon^2, $$

hence $R^2\_{2\,\text{pred}} \ge R^2\_{1\,\text{pred}}$, with equality only when the extra term vanishes: either $\beta_2=0$ ($X_2$ has no effect on $Y$) or $\text{var}(X_2\mid X_1)=0$ ($X_2$ is already determined by $X_1$). So adding predictors either improves the fit or leaves it unchanged &mdash; never worse.


<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Reading a two-predictor fit

A regression of monthly rent $Y$ (in dollars) on floor area $X_1$ (in square feet) and distance from campus $X_2$ (in miles) gives $\hat\beta_0=400$, $\hat\beta_1=1.2$, $\hat\beta_2=-90$, and $R^2=0.62$.

<ol type="a">
  <li>State the interpretation of $\hat\beta_1$ in one sentence, being explicit about what is held fixed.</li>
  <li>Predict the rent of a $700$ sq ft apartment $1.5$ miles from campus.</li>
  <li>What does $R^2=0.62$ say, and what does it <em>not</em> say?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Dropping a predictor

For a binary predictor $X_1$, the two-predictor fit gives $\beta_1=2$ and $\beta_2=5$, and the predictors satisfy $E[X_2\mid X_1{=}1]-E[X_2\mid X_1{=}0]=-0.4$.

<ol type="a">
  <li>Compute the single-predictor coefficient $\beta_1^{\prime}$ of $X_1$ alone.</li>
  <li>Someone who only fits $X_1$ concludes that $X_1$ is unrelated to $Y$. In one sentence, say what they have missed.</li>
  <li>What would $\beta_1^{\prime}$ be if $X_1$ and $X_2$ were independent?</li>
</ol>
</div>

</details>

## 4.3 Covariance matrix and estimation of regression coefficients {#sec-4-3}

Here we derive formulas for the regression coefficients in terms of the covariances between predictors and between predictors and the response. This (1) clarifies the relationship between single- and multi-predictor coefficients, and (2) gives us an estimator for the multi-predictor case. We're generalizing the single-predictor relationship $\text{cov}(X,Y)=\beta_1\sigma_X^2$.

Consider a two-predictor model, and set $\beta_0=E[X_1]=E[X_2]=0$ for simplicity (these cancel out in the end &mdash; check for yourself that everything still works when they're not zero!). Then $\text{cov}(X_1,Y)=E[X_1Y]$, and just as in the single-predictor case,

$$
\begin{aligned}
\text{cov}(X_1,Y)
&= E[X_1Y] \\
&= E[X_1E[Y\mid X_1]] \\
&= E[X_1(\beta_1X_1+\beta_2X_2)] \\
&= \beta_1E[X_1^2]+\beta_2E[X_1X_2] \\
&= \beta_1\sigma_{X_1}^2 + \beta_2\,\text{cov}(X_1,X_2).
\end{aligned}
$$

Doing the same for $X_2$ gives two equations:

$$ \text{cov}(X_1,Y) = \beta_1\sigma_{X_1}^2 + \beta_2\,\text{cov}(X_1,X_2), \qquad \text{cov}(X_2,Y) = \beta_2\sigma_{X_2}^2 + \beta_1\,\text{cov}(X_1,X_2). $$

As in the single-predictor case, it's useful to write $\beta_1,\beta_2$ as expectations we can estimate by averaging over data. These two equations can be rewritten with the <span class="term">[covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix)</span>

$$ \Sigma = \begin{bmatrix} \sigma_{X_1}^2 & \text{cov}(X_1,X_2) \\ \text{cov}(X_1,X_2) & \sigma_{X_2}^2 \end{bmatrix}, $$

as

$$ \begin{bmatrix} \text{cov}(X_1,Y) \\ \text{cov}(X_2,Y) \end{bmatrix} = \Sigma \begin{bmatrix} \beta_1 \\ \beta_2 \end{bmatrix}. $$

This is consistent with the formula above: dividing the first equation by $\text{var}(X_1)$ recovers $\beta_1^{\prime} = \beta_1+\beta_2\beta_{1,2}$, where $\beta_{1,2}=\text{cov}(X_1,X_2)/\text{var}(X_1)$ is the (single-predictor) regression coefficient of $X_1$ with $X_2$ as the response. This generalizes to many predictors: in general, $\Sigma$ is a $K\times K$ matrix with entries $\Sigma_{i,j}=\text{cov}(X_i,X_j)$.

Letting $\Sigma^{-1}$ denote the matrix inverse ($\Sigma^{-1}\Sigma=I$), we can solve for the coefficients:

$$ \begin{bmatrix} \beta_1 \\ \beta_2 \end{bmatrix} = \Sigma^{-1}\begin{bmatrix} \text{cov}(X_1,Y) \\ \text{cov}(X_2,Y) \end{bmatrix} = \frac{1}{\sigma_{X_1}^2\sigma_{X_2}^2-\text{cov}(X_1,X_2)^2} \begin{bmatrix} \sigma_{X_2}^2 & -\text{cov}(X_1,X_2) \\ -\text{cov}(X_1,X_2) & \sigma_{X_1}^2 \end{bmatrix} \begin{bmatrix} \text{cov}(X_1,Y) \\ \text{cov}(X_2,Y) \end{bmatrix}. $$

Compare this to the single-predictor formula $\beta_1^{\prime} = \text{cov}(X,Y)/\text{var}(X)$: here $\Sigma^{-1}$ plays the role of $1/\text{var}(X)$, and the vector of $\beta$s and $Y$-$X$ covariances play the roles of $\beta_1$ and $\text{cov}(X,Y)$. If $\Sigma=I$ (uncorrelated, unit-variance predictors), each $\beta_i$ can be solved for separately, recovering the single-predictor formulas.

We don't usually bother with closed-form solutions once there are many predictors &mdash; they get complicated fast. But for two predictors,

$$ \beta_1 = \frac{\text{cov}(X_1,Y)\sigma_{X_2}^2 - \text{cov}(X_2,Y)\,\text{cov}(X_1,X_2)}{\sigma_{X_1}^2\sigma_{X_2}^2 - \text{cov}(X_1,X_2)^2}, $$

which, if all variances are set to one, becomes

$$ \beta_1 = \frac{1}{1-\rho_{1,2}^2}\big(\rho_1-\rho_{1,2}\rho_2\big), $$

where $\rho_i$ is the correlation coefficient between $X_i$ and $Y$, and $\rho_{1,2}$ is the correlation coefficient between $X_1,X_2$. If $X_1,X_2$ are uncorrelated ($\rho_{1,2}=0$), this reduces to the usual single-predictor connection.

<div class="example" markdown="1">
#### Example (correlated predictors)

Consider the model

$$
X_1 \sim \text{Normal}(0,1), \qquad
X_2\mid X_1 \sim \text{Normal}(bX_1,\,1-b^2),
$$

$$
Y\mid(X_1,X_2) \sim \text{Normal}(\beta_1X_1+\beta_2X_2,\,\sigma^2),
$$

with $b\in[0,1]$. Note $b=\beta_{1,2}$: the single-predictor regression coefficient of $X_1$ with $X_2$ as the response, which here (both predictors having unit variance) is also the correlation coefficient.

<u>Question:</u>
<ol type="a">
<li>Show that $\text{var}(X_1)=\text{var}(X_2)=1$ and $\text{cov}(X_1,X_2)=b$.</li>
<li>Express $\beta_1$ in terms of $b$, $\text{cov}(X_1,Y)$, and $\text{cov}(X_2,Y)$.</li>
<li>Find the single-predictor coefficient $\beta_1^{\prime}$ when only $X_1$ is used.</li>
</ol>

<u>Solution:</u>
<ol type="a">
<li>By definition $\text{var}(X_1)=1$, and $\text{var}(X_2)=b^2\text{var}(X_1)+(1-b^2)=1$; also $\text{cov}(X_1,X_2)=b\,\text{var}(X_1)=b$.</li>
<li>Substituting into the formula above, $\beta_1 = \dfrac{\text{cov}(X_1,Y)-\text{cov}(X_2,Y)b}{1-b^2}$.</li>
<li>The single-predictor slope of $Y$ on $X_1$ is $\beta_1+b\beta_2$.</li>
</ol>
</div>

### Estimating regression coefficients

Now suppose we have data $\lbrace(X_{i,1},\dots,X_{i,K},Y_i)\rbrace_{i=1}^N$. Form the <span class="term">[design matrix](https://en.wikipedia.org/wiki/Design_matrix)</span> with samples in rows and predictors in columns,

$$ X = \begin{bmatrix} X_{1,1} & \cdots & X_{1,K} \\ \vdots & \ddots & \vdots \\ X_{N,1} & \cdots & X_{N,K} \end{bmatrix} \in \mathbb R^{N\times K}, \qquad Y = \begin{bmatrix} Y_1 \\ \vdots \\ Y_N \end{bmatrix} \in \mathbb R^N. $$

Assuming $X,Y$ are centered (no intercept), the empirical covariance quantities are

$$ \hat\Sigma = \frac1N X^\top X \in \mathbb R^{K\times K}, \qquad \hat c = \frac1N X^\top Y \in \mathbb R^K. $$

When $\hat\Sigma$ is invertible, this gives the closed-form <span class="term">[ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)</span> (OLS) estimator,

$$ \hat\beta = \hat\Sigma^{-1}\hat c = (X^\top X)^{-1}X^\top Y. $$

<details class="optional-section" open markdown="1">
<summary><strong>Aside: the Moore&ndash;Penrose pseudoinverse</strong> <span class="optional-badge">(optional)</span></summary>

If $\hat\Sigma$ is singular (e.g. under exact collinearity, or if $K>N$), we instead use the Moore&ndash;Penrose pseudoinverse,

$$ \hat\beta = X^{+}Y, \qquad X^{+} = V\Sigma^{+}U^\top \quad\text{for } X=U\Sigma V^\top \text{ (the SVD of } X), $$

where $\Sigma^{+}$ inverts the nonzero singular values and leaves the zero ones unchanged.

</details>


<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Solving the covariance system

Suppose $\sigma_{X_1}^2=\sigma_{X_2}^2=1$, $\text{cov}(X_1,X_2)=0.5$, $\text{cov}(X_1,Y)=1$, and $\text{cov}(X_2,Y)=0.5$.

<ol type="a">
  <li>Write out $\Sigma$ and solve $\Sigma\beta = (\text{cov}(X_1,Y),\text{cov}(X_2,Y))^\top$ for $\beta_1,\beta_2$.</li>
  <li>Compute the single-predictor coefficient $\beta_1^{\prime}=\text{cov}(X_1,Y)/\sigma_{X_1}^2$.</li>
  <li>Here $\beta_1^{\prime}=\beta_1$ even though the predictors are correlated. Which term in $\beta_1^{\prime}=\beta_1+\beta_2\beta_{1,2}$ explains that?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Reading a statsmodels fit

Consider the following <code>statsmodels</code> fit:

```python
import statsmodels.api as sm
X_design = sm.add_constant(X)
fit = sm.OLS(Y, X_design).fit()
beta = fit.params
```

<ol type="a">
  <li>What column does <code>sm.add_constant(X)</code> add, and which regression coefficient does it represent?</li>
  <li>What does <code>fit.params</code> contain?</li>
  <li>Use <code>fit.summary()</code> to identify the estimated coefficient, standard error, and $p$-value for each predictor.</li>
</ol>
</div>

</details>

## 4.4 Sample distribution and collinearity {#sec-4-4}

Just as before, we want to understand the sample distribution of the coefficients &mdash; but now we need the *joint* distribution of $(\hat\beta_1,\hat\beta_2,\dots,\hat\beta_K)$. We'll focus on the two-predictor case. In the demo below, each faint line at the top is a least-squares fit to a fresh replicate of the same $X$ values, so the spread of those lines is a visual proxy for the width of the sample distribution of $\hat\beta_1$. The cloud underneath is the joint sample distribution of $(\hat\beta_1,\hat\beta_2)$: as the predictor correlation $b$ approaches $\pm1$ it stretches along the line $\hat\beta_1+\hat\beta_2=\text{const}$, so the individual coefficients become poorly determined even though their sum does not.

{% include_relative demos/sample_dist.html %}

To build intuition, imagine $X_1,X_2$ are very highly correlated (if perfectly correlated, we say they're <span class="term">[collinear](https://en.wikipedia.org/wiki/Multicollinearity)</span>). Then

$$ Y = \beta_1X_1+\beta_2X_2+\epsilon \approx \beta_1X_1+\beta_2X_1+\epsilon = (\beta_1+\beta_2)X_1+\epsilon. $$

There are many ways to choose $\beta_1,\beta_2$ so the surface $\beta_1x_1+\beta_2x_2$ stays close to the data, since a change in $\beta_1$ can be compensated by an opposite change in $\beta_2$. **This means that if we estimate $\hat\beta_1,\hat\beta_2$ and then regenerate new data, we could get very different values of $\hat\beta_1,\hat\beta_2$, so long as $\hat\beta_1+\hat\beta_2$ stays close to what we got before.** The demo below refits on a fresh dataset each time and shows the same fitted surfaces from two directions: along $u=(X_1+X_2)/\sqrt2$, the direction in which the predictors move together, and along $v=(X_1-X_2)/\sqrt2$, the direction in which they disagree. Raise the correlation and the first view stays a tight bundle while the second fans out.

{% include_relative demos/sample_dist2.html %}

### Sample distribution formula

It can be shown that, assuming the covariance matrix $\Sigma$ is known, $\hat\beta$ has a multivariate Normal distribution:

$$ \hat\beta \sim \operatorname{Normal}_K\Big(\beta,\ \frac{\sigma_\epsilon^2}{N}\Sigma^{-1}\Big). $$

If $\Sigma=I$, this reduces to the sample distribution we found in the single-predictor case.


<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Correlation and standard errors

Take two standardized predictors with correlation $\rho$, so

$$ \Sigma = \begin{bmatrix}1 & \rho \\ \rho & 1\end{bmatrix}. $$

Use $\hat\beta \sim \operatorname{Normal}\_2\big(\beta,\ (\sigma\_\epsilon^2/N)\Sigma^{-1}\big)$ with $\sigma_\epsilon^2=1$ and $N=100$.

<ol type="a">
  <li>Write $\text{se}(\hat\beta_1)$ as a function of $\rho$.</li>
  <li>Evaluate it at $\rho=0$ and at $\rho=0.9$.</li>
  <li>What is the sign of the correlation between $\hat\beta_1$ and $\hat\beta_2$ when $\rho>0$, and what does that mean about how the two estimates move together across replicates?</li>
</ol>
</div>

</details>

## 4.5 Categorical predictors {#sec-4-5}

Multiple-predictor models often arise when predicting $Y$ from a categorical predictor, like race. In this case, we need to convert the categories to numerical values. If there are two categories (e.g. YES/NO), we map them to $0$ or $1$. With $3$ categories (e.g. White, Black, Other), we might think to map them to $0,1,2$ &mdash; but this has a problem: a change from $1$ to $2$ shouldn't necessarily be "the same size" as a change from $0$ to $1$. **There's no natural ordering of the $x$ values.** We call such predictors <span class="term">[qualitative](https://en.wikipedia.org/wiki/Categorical_variable)</span> rather than <span class="term">[quantitative](https://en.wikipedia.org/wiki/Continuous_or_discrete_variable)</span>, since they express a quality of the data point rather than a numerical quantity.

To handle this, we create <span class="term">[dummy variables](https://en.wikipedia.org/wiki/Dummy_variable_(statistics))</span>: a set of indicator variables, one per category (minus one, as we'll see). In Python, `pd.get_dummies` does this conversion.

<div class="example" markdown="1">
#### Example (racial disparities in earnings)

We'll fit the earnings data to a model with race as a predictor: what's the association between race and earnings among US adults? We could use a single binary predictor (e.g. White vs. non-White), but that's limiting. Instead, we make one dummy variable per race category. The dataset has $4$ categories, $\lbrace\text{Black},\text{White},\text{Hispanic},\text{Other}\rbrace$. In principle we could create a binary variable for each:

$$ Y = \beta_0 + \beta_{\text{black}}X_{\text{black}}+\beta_{\text{hispanic}}X_{\text{hispanic}}+\beta_{\text{other}}X_{\text{other}}+\beta_{\text{white}}X_{\text{white}}+\epsilon, $$

but this is problematic: at least one of these predictors must equal $1$ for every observation, so they're perfectly collinear. By default, Python drops the first category (alphabetically), giving

$$ Y = \beta_0+\beta_{\text{hispanic}}X_{\text{hispanic}}+\beta_{\text{other}}X_{\text{other}}+\beta_{\text{white}}X_{\text{white}}+\epsilon. $$

<u>Question:</u> fit this model. What's the expected earnings disparity between someone who is White and someone who is Hispanic?

<u>Solution:</u> see the companion Colab notebook. The regression coefficients, in terms of conditional expectations, are

$$
\begin{aligned}
\beta_{\text{white}}
&= E[Y\mid X_{\text{white}}{=}1,X_{\text{hispanic}}{=}X_{\text{other}}{=}0] \\
&\quad - E[Y\mid X_{\text{white}}{=}0,X_{\text{hispanic}}{=}X_{\text{other}}{=}0] \\
&= E[Y\mid\text{White}]-E[Y\mid\text{Black}] \approx 4.9,
\end{aligned}
$$

$$
\begin{aligned}
\beta_{\text{hispanic}}
&= E[Y\mid X_{\text{hispanic}}{=}1,X_{\text{white}}{=}X_{\text{other}}{=}0] \\
&\quad - E[Y\mid X_{\text{hispanic}}{=}0,X_{\text{white}}{=}X_{\text{other}}{=}0] \\
&= E[Y\mid\text{Hispanic}]-E[Y\mid\text{Black}] \approx -0.7.
\end{aligned}
$$

Our goal is $E[Y\mid\text{White}]-E[Y\mid\text{Hispanic}]$, which we can get by subtracting these two:

$$ E[Y\mid\text{White}]-E[Y\mid\text{Hispanic}] = \big(\beta_0+\beta_{\text{white}}\big) - \big(\beta_0+\beta_{\text{hispanic}}\big) = \beta_{\text{white}}-\beta_{\text{hispanic}}. $$
</div>


<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Counting dummy variables

A predictor records the season a measurement was taken: winter, spring, summer, or fall.

<ol type="a">
  <li>How many dummy variables go into the model, and why not one per category?</li>
  <li>With winter dropped as the baseline, state what $\beta_{\text{summer}}$ means.</li>
  <li>Using the fitted coefficients, how would you compute $E[Y\mid\text{summer}]-E[Y\mid\text{fall}]$?</li>
</ol>
</div>

</details>

## Problems {#problems}

<div class="exercise" markdown="1">
#### Problem 4.1 &mdash; A binary and Normal predictor

Consider the linear regression model

$$ Y\mid(X_1,X_2) \sim \text{Normal}(\beta_0+\beta_1X_1+\beta_2X_2,\ \sigma^2), $$

where the two predictors satisfy $X_1 \sim \text{Bernoulli}(q)$ and $X_2\mid X_1 \sim \text{Normal}(bX_1,s^2)$.

<ol type="a">
<li>Give at least two real-world examples where this would be a reasonable model relating three variables $X_1,X_2,Y$.</li>
<li>What are $\text{cov}(X_1,X_2)$ and $\text{var}(X_2)$ in terms of the model parameters $q,b,s,\beta_0,\beta_1,\beta_2,\sigma^2$? You may assume $\beta_0=0$ for this part and the rest of the exercise &mdash; it simplifies the calculations without changing the results.</li>
<li>Derive a formula for $\text{cov}(Y,X_1)$ in terms of $\beta_1,q,\beta_2,b$.</li>
<li>How does the formula from part (c) relate to the formula for $\text{cov}(Y,X_1)$ in the single-predictor regression model (Unit 3)? For what parameter values do the two formulas coincide? This is a special case of the general relationship between $\beta_1$ and covariances that we saw in class.</li>
<li>Derive the formula
$$ \text{var}(Y) = q(1-q)\big(\beta_1^2+\beta_2^2b^2+2\beta_1\beta_2b\big) + \beta_2^2s^2 + \sigma^2. $$
You'll need the formula for the variance of a sum of two (not necessarily independent) random variables &mdash; see the "addition and multiplication" section of the <a href="https://en.wikipedia.org/wiki/Variance#Properties">Wikipedia page on variance</a>.</li>
<li>The calculation in part (c) lets us answer a classic textbook question, in the more restrictive context of a binary and Normal predictor: is it possible for $\beta_1$ and $\beta_2$ to <em>both be negative</em>, yet the marginal slope of $Y$ vs. $X_1$ be <em>positive</em>? If so, generate simulated data demonstrating this.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 4.2 &mdash; Earnings data

Consider the earnings data, loaded with

```python
url = (
    "https://raw.githubusercontent.com/avehtari/"
    "ROS-Examples/master/Earnings/data/earnings.csv"
)
df = pd.read_csv(url)
```

As in Unit 3's problems, you'll study the association between earnings and gender, but now with multiple predictors.

<ol type="a">
<li>Perform a linear regression using <code>statsmodels</code> with gender and height as predictors.</li>
<li>Interpret each regression coefficient (as we did in class for the test-score example).</li>
<li>Based on your analysis, which factor &mdash; height or gender &mdash; is more important?</li>
<li>Using the fitted model, predict the chance that someone who is not male and is $5.8$ft tall earns more than a male of the same height. To gauge how much height matters, compare this to the chance a male earns more than a non-male, regardless of height.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 4.3 &mdash; Sample distribution

In class we wrote code to generate samples from the sample distribution of $(\hat\beta_1,\hat\beta_2)$ in the model

$$
X_1 \sim \text{Normal}(0,1), \qquad
X_2\mid X_1 \sim \text{Normal}(bX_1,\,1-b^2),
$$

$$
Y\mid(X_1,X_2) \sim \text{Normal}(\beta_1X_1+\beta_2X_2,\,\sigma^2),
$$

with a function that takes $\beta_1,\beta_2,\beta_0$ as inputs and returns a dataframe of simulated $\hat\beta_1,\hat\beta_2$ values. When we plotted the estimated correlation coefficient between $\hat\beta_1$ and $\hat\beta_2$ as a function of $b$, it was a decreasing line.

<ol type="a">
<li>What would happen if, instead, we plotted $\text{se}(\hat\beta_1)$ as a function of $b$? Would it increase, decrease, or neither? (Both $X_1,X_2$ are standardized, so the marginal distribution of $X_1$ doesn't change as $b$ varies.) You can give a geometric argument or a calculation, but check your answer with simulation either way.</li>
<li>Is it possible to have large standard errors on all the $\hat\beta_i$ (relative to their true values), yet still have $R^2$ close to one? If so, for what parameter values does this happen? Support your answer with simulation.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 4.4 &mdash; Regression on a downstream predictor

Suppose we have a large amount of data from the model

$$ X_1 \sim \text{Normal}(0,1), \qquad X_2\mid X_1 \sim \text{Normal}(3X_1,1), \qquad Y\mid(X_1,X_2) \sim \text{Normal}(X_1-2X_2,\,1). $$

If we perform a single-predictor regression of $Y$ on $X_2$ alone, what is the regression coefficient $\beta_2^{\prime}$?
</div>

<div class="exercise" markdown="1">
#### Problem 4.5 &mdash; Simpson's paradox with a parameter

Consider two binary predictors $X_1,X_2\in\lbrace0,1\rbrace$ with joint distribution

<table class="prob-table">
<tr><th>$P(X_1,X_2)$</th><th>$X_2=0$</th><th>$X_2=1$</th></tr>
<tr><th>$X_1=0$</th><td>0.4</td><td>0.1</td></tr>
<tr><th>$X_1=1$</th><td>0.1</td><td>0.4</td></tr>
</table>

and suppose $Y\mid(X_1,X_2) = X_1+cX_2+\epsilon$.

<ol type="a">
<li>Compute the single-predictor regression coefficient $\beta_1^{\prime}$ (in terms of $c$).</li>
<li>Compute the single-predictor regression coefficient $\beta_2^{\prime}$ (in terms of $c$).</li>
<li>For which values of $c$ does the model exhibit Simpson's paradox?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 4.6 &mdash; Bivariate Normal predictors

Let $X=(X_1,X_2)^\top$ be a bivariate Normal vector with mean zero and covariance matrix

$$ \Sigma = \begin{bmatrix} 10 & 2 \\ 2 & 10 \end{bmatrix}. $$

Suppose $Y = 0.5X_1-1.5X_2+\epsilon$, $\epsilon\sim\text{Normal}(0,1)$.

<ol type="a">
<li>Compute the single-predictor regression coefficient $\beta_1^{\prime}$ for $Y$ on $X_1$.</li>
<li>Does Simpson's paradox occur in this setup? Explain.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 4.7 &mdash; Endurance performance

A sports scientist studies the relationship between athletes' endurance performance and two predictors: weekly training hours ($X_1$) and VO2 max ($X_2$, in ml/kg/min), on a 10km run time $Y$ (in minutes), using data from $100$ athletes.

A single-predictor regression of $Y$ on $X_1$ alone gives:

```text
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.8884      0.285      3.114      0.002       0.322       1.454
x1             1.8436      0.284      6.487      0.000       1.280       2.408
==============================================================================
```

Adding VO2 max ($X_2$) as a second predictor gives:

```text
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.0170      0.011     94.148      0.000       0.996       1.038
x1             1.0027      0.011     89.354      0.000       0.980       1.025
x2             3.0020      0.011    261.492      0.000       2.979       3.025
==============================================================================
```

What is the regression coefficient $\beta_{1,2}$ of $X_1$ with $X_2$ as the response variable?
</div>

<div class="exercise" markdown="1">
#### Problem 4.8 &mdash; Omitted-variable bias

A researcher fits a model with two predictors $X_1,X_2$, but the available dataset only contains $X_1$. The **true** (long) model is

$$ Y = 1.0+2.0X_1+4.0X_2+\epsilon_1, $$

and the (unobserved) relationship between the predictors is $X_2 = 0.5+0.5X_1+\epsilon_2$, with $E[\epsilon_1]=E[\epsilon_2]=0$.

The researcher instead runs the regression $Y = \beta_0+\hat\beta_1X_1+\epsilon_1^{\prime}$ using only $X_1$.

<u>Question:</u> what is the expected value (approximately) of the coefficient $\hat\beta_1$ the researcher will estimate?
</div>
