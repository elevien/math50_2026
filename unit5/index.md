---
layout: notes
title: "Unit 5: Nonlinear Models & the Bias-Variance Tradeoff"
unit_title: "Unit 5"
unit_subtitle: "Interactions, feature maps, cross-validation, overfitting, and Fourier features"
toc:
  - {href: "#sec-5-1", label: "5.1 Interaction terms"}
  - {href: "#sec-5-2", label: "5.2 Feature maps for nonlinear models"}
  - {href: "#sec-5-3", label: "5.3 Cross-validation & the bias-variance tradeoff"}
  - {href: "#sec-5-4", label: "5.4 Orthogonality & Fourier features"}
  - {href: "#problems", label: "Problems"}
---

<div class="unit-overview" markdown="1">

In this unit, we begin the study of **nonlinear** model. This is one route to building a bridge between machine learning and statistics. The key is that even if the model is nonlinear in the predictors, we can still use the linear regression framework if it is linear in the parameters. The first nonlinear models we look at are those with **interactions**. Interactions occur when the effect of one predictor depends on the value of another, and we can include them by adding products of predictors to our model. Interactions are examples of **features** &mdash; that is, new predictors built from functions of the old ones. With more complex models, come new problems, the most important one is **overfitting**. We will understand this in terms of the bias variance tradeoff. 

#### Concepts

Interactions, residual plots, other nonlinear models, cross-validation, training and test error, overfitting and the bias-variance tradeoff, orthogonal features, the discrete Fourier transform and the periodogram.

#### Things to practice

- Use residual plots to identify nonlinearities in a model.
- Build nonlinear models by making new features of old ones. 
- Interpret results of cross validation
- Understand the distinction between training and test error.
- Explain intuitively the idea behind the bias-variance tradeoff in predictions

</div>

<p class="pdf-link"><a href="unit5.pdf">Unit 5 notes pdf</a> <a href="https://colab.research.google.com/github/elevien/math50_2026/blob/main/unit5/unit5.ipynb">Unit 5 notebook</a></p>

## 5.1 Interaction terms {#sec-5-1}

The key assumption of the multiple-predictor regression models we've seen so far is that the "effect" of one predictor doesn't depend on the value of another. Here are some examples where that could fail:

- The difference in test scores between kids whose mothers did and didn't attend high school might depend on the mother's IQ.
- The association between earnings and height might depend on gender &mdash; e.g. being taller tends to give men a bigger advantage than women.
- The effect of a drug for treating an infection might depend on whether someone has had it before.

We call dependencies like these <span class="term">[interactions](https://en.wikipedia.org/wiki/Interaction_(statistics))</span>. It turns out we can include interactions within the regression framework we already have, as the following makes concrete. Let's work with two predictors (if it helps, think of $X_1=X_{\text{hs}}$ and $X_2=X_{\text{iq}}$, as in the test-score example from Unit 4). In each case above, what's going on is that the slope of $E[Y\mid X_1,X_2]$ vs. $X_1$ for fixed $X_2$ isn't a constant $\beta_1$, but something that depends on $X_2$. The simplest way to capture this is to let the effect of $X_2$ on the slope be *linear* in $X_2$: instead of $\beta_1$ being the slope, we'd have $\beta_1+J_{1,2}X_2$ be the slope. This gives a regression model with a nonlinear term:

$$
\begin{aligned}
Y &= \beta_0+(\beta_1+J_{1,2}X_2)X_1+\beta_2X_2+\epsilon\\
&= \beta_0+\beta_1X_1+\beta_2X_2+J_{1,2}X_1X_2+\epsilon.
\end{aligned}
$$

We call $X_1X_2$ an <span class="term">[interaction](https://en.wikipedia.org/wiki/Interaction_(statistics))</span> term &mdash; but this isn't linear anymore, and the coefficients don't have the same interpretation as before. How do we deal with this? The idea: create a *new predictor* from the existing ones that accounts for the interaction. Writing the plain two-predictor model as

$$ Y = \alpha_0+\alpha_1X_1+\alpha_2X_2+\epsilon $$

(writing $\alpha$ for its coefficients to distinguish it from the expanded model), the new predictor that accounts for the interaction is $X_3=X_1X_2$. Note $X_3$ will be correlated with both $X_1$ and $X_2$, but not *perfectly* correlated unless $X_1,X_2$ are perfectly correlated with each other. The new model is

$$ Y\mid X \sim \text{Normal}(\beta_0+\beta_1X_1+\beta_2X_2+J_{1,2}X_3,\ \sigma^2). $$

How do we interpret the coefficients now? Using the usual formula for a regression coefficient,

$$
\begin{aligned}
E[Y\mid X_1{=}x{+}1,X_2] - E[Y\mid X_1{=}x,X_2] &= \big(\beta_0+\beta_1(x{+}1)+\beta_2X_2+J_{1,2}(x{+}1)X_2\big)\\
&\quad - \big(\beta_0+\beta_1x+\beta_2X_2+J_{1,2}xX_2\big)\\
&= \beta_1+J_{1,2}X_2,
\end{aligned}
$$

(replacing $X_3$ with $X_1X_2$). In other words, $J_{1,2}$ is the additional change in the slope between $Y$ and $X_1$, with all other predictors fixed, per unit change in $X_2$. If $J_{1,2}$ is positive (negative), increasing $X_2$ tends to increase (decrease) the difference between the conditional averages of $Y$ &mdash; the new predictor $X_3=X_1X_2$ captures an interaction where the association between $X_1$ and $Y$ depends on $X_2$. Doing the same calculation the other way,

$$ E[Y\mid X_1,X_2{=}x{+}1] - E[Y\mid X_1,X_2{=}x] = \beta_2+J_{1,2}X_1. $$

<div class="example" markdown="1">
#### Example (visualizing interactions)

Consider the model above with interaction term $X_3=X_1X_2$, where $X_1\sim\text{Normal}(0,1)$, $X_2\sim\text{Bernoulli}(1/2)$, and $\beta_1=1.2,\beta_2=0,J_{1,2}=-1$.

<u>Question:</u> what are the slopes of $E[Y\mid X_1,X_2{=}0]$ vs. $X_1$, and of $E[Y\mid X_1,X_2{=}1]$ vs. $X_1$?

<u>Solution:</u> the slope vs. $X_1$ is $\beta_1+J_{1,2}X_2$: $\beta_1=1.2$ when $X_2=0$ and $\beta_1+J_{1,2}=0.2$ when $X_2=1$. The lines are not parallel, which is what an interaction looks like (simulated data with $\beta_0=0$ and $\sigma=0.5$):

<img src="{{ '/unit5/fig/interaction_example.svg' | relative_url }}" alt="Scatter plot of Y vs. X1 for simulated data. Points with X2 = 0 (orange circles) follow a steep line of slope 1.2; points with X2 = 1 (teal plus signs) follow a nearly flat line of slope 0.2. The two lines cross at X1 = 0." style="width: 100%; max-width: 520px;">
</div>

<div class="example" markdown="1">
#### Example (interaction in the test-score model)

Returning to the test-score data with high-school education and IQ as predictors, we fit

$$ Y = \beta_0+\beta_{\text{hs}}X_{\text{hs}}+\beta_{\text{iq}}X_{\text{iq}}+J_{\text{hs},\text{iq}}X_{\text{hs}}X_{\text{iq}}. $$

<u>Question:</u> what are the values and interpretations of the coefficients? What do the results tell us about the "effect" of IQ on test scores, and how it relates to high-school education?

<u>Solution:</u> `statsmodels` gives:

```text
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        -11.4820     13.758     -0.835      0.404     -38.523      15.559
mom_hs        51.2682     15.338      3.343      0.001      21.122      81.414
mom_iq         0.9689      0.148      6.531      0.000       0.677       1.260
hs_iq         -0.4843      0.162     -2.985      0.003      -0.803      -0.165
```

<ul>
<li>$\beta_0 \approx -11$: no clear interpretation, since zero IQ doesn't make sense.</li>
<li>$\beta_{\text{iq}} \approx 1$: the expected signed difference in scores between students whose mothers' IQ differs by $1$ point, among students whose mothers did <em>not</em> attend high school.</li>
<li>$\beta_{\text{hs}} \approx 51$: the expected difference between scores of students whose mothers did and didn't attend high school, when IQ is zero &mdash; much like $\beta_0$, this doesn't really make sense, so $\beta_{\text{hs}}$ doesn't have a clean interpretation on its own.</li>
<li>$J_{\text{hs},\text{iq}} \approx -0.5$: the expected change, between the high-school and non-high-school groups, in the difference between test scores of students whose mothers' IQ differs by one point &mdash; i.e. the change in the regression slope of scores vs. IQ between the two groups. It's easier to interpret $\beta_{\text{iq}}+J_{\text{hs},\text{iq}}$, the expected signed difference in scores per IQ point <em>among students whose mothers did attend high school</em> (the analogue of $\beta_{\text{iq}}$ for that group).</li>
<li>The association between IQ and test scores appears weaker among students whose mothers attended high school. This makes intuitive sense: if someone attended high school, the subtle cognitive differences that IQ is meant to measure may be less relevant.</li>
</ul>
</div>

The example above shows that adding an interaction term can make the interpretation of the coefficients a bit clunky &mdash; even for coefficients not directly involved in the interaction. In particular, $\beta_0$ and $\beta_{\text{hs}}$ describe mothers with IQ zero, so they have no useful interpretation. To fix this, define the centered predictor

$$ \Delta_{\text{iq}} = X_{\text{iq}} - \overline X_{\text{iq}}, $$

and fit

$$ Y = \beta_0+\beta_{\text{hs}}X_{\text{hs}}+\beta_{\text{iq}}\Delta_{\text{iq}}+J_{\text{hs},\text{iq}}X_{\text{hs}}\Delta_{\text{iq}}+\epsilon. $$

Now $\Delta_{\text{iq}}=0$ means a mother of *average* IQ, so

- $\beta_0$ is the expected score of a student whose mother did not attend high school and has average IQ;
- $\beta_{\text{hs}}$ is the expected difference between the scores of students whose mothers did and didn't attend high school, when the mothers have average IQ.

Centering only shifts where we measure IQ from, so $\beta_{\text{iq}}$ and $J_{\text{hs},\text{iq}}$ (and their standard errors) are the same as before: $\beta_{\text{iq}}$ is still the slope of score vs. IQ among students whose mothers did not attend high school. You can also standardize a predictor to get around this awkwardness.

### Residual plots

How do we identify when it's appropriate to add an interaction term? With two predictors, we can visualize the data directly by plotting $Y$ vs. $X_1$ slopes for different $X_2$ values (especially when one predictor is binary). With many predictors, it's less clear what plot might reveal a hidden interaction &mdash; we could add every possible interaction term, but this quickly becomes impractical and leads to <span class="term">[overfitting](https://en.wikipedia.org/wiki/Overfitting)</span> (more on this soon).

The basic idea of <span class="term">[residual](https://en.wikipedia.org/wiki/Residual_(statistics)) plots</span>: plot the difference between the observed $y$ values and the model's prediction,

$$ \hat\epsilon_j = Y_j - \hat\beta_0 - \sum_{k=1}^K \hat\beta_kX_{k,j}, $$

(note the hat: $\hat\epsilon_j$ is the *residual* we can compute from the fitted model, as opposed to the unobservable noise $\epsilon_j$ in the model itself) in a way that reveals disagreement between the model and the data. With a single predictor, we can just plot $\hat\epsilon_j$ vs. $X$ &mdash; if the residuals don't look Normal, or their variance/mean changes with $X$, we should be skeptical of the model. **With multiple predictors, what do we plot on the horizontal axis?** The answer: plot $\hat\epsilon_j$ against the *fitted value* $\hat y_j=\hat\beta_0+\sum_k\hat\beta_kX_{k,j}$, not against any single predictor (and, importantly, not against $Y_j$ itself). The demo below puts the two side by side for a two-predictor model, with the least-squares trend drawn through each cloud. On the left, against $\hat y_j$, the points form a random horizontal band and the trend stays flat however you move the sliders; on the right, against $Y_j$, the same residuals tilt.

{% include_relative demos/res.html %}

<div class="example" markdown="1">
#### Example (residual plots)

Consider a two-predictor model with true parameters $\beta_0=0,\beta_1=0.2,\beta_2=20,\sigma=1$.

<u>Question:</u> compare two ways of plotting the residuals: (a) $\hat\epsilon_j$ vs. $\hat y_j$, and (b) $\hat\epsilon_j$ vs. $Y_j$. Why does (b) show an apparent bias toward high $\hat\epsilon_j$ for large $Y_j$ that's absent from (a)?

<u>Solution:</u> see the companion Colab notebook.
</div>

To see why, recall

$$ \hat\epsilon_j \approx Y_j - \underbrace{E[Y\mid(X_1,\dots,X_K)=(X_{1,j},\dots,X_{K,j})]}_{\approx\ \hat\beta_0+\sum_k\hat\beta_kX_{k,j}}, $$

so $\hat\epsilon_j$ is approximately $\text{Normal}(0,\sigma^2)$. This tells us how the points should be spread in the *vertical* direction &mdash; it says nothing about their spread in the horizontal direction, which is determined by the distribution of the predictors. So we expect a plot that's symmetric around $\hat\epsilon_j=0$ for every value of $\hat y_j$, but any spread horizontally is fine. Now compare plotting $Y_j$ on the horizontal axis instead of $\hat y_j$: by the approximation above, $\hat\epsilon_j$ and $Y_j$ are correlated (in fact $\text{cov}(\hat\epsilon_j,Y_j)=\text{var}(\hat\epsilon_j)>0$ always, by the orthogonality of least-squares residuals to the fitted values) &mdash; which is exactly the bias we see for small/large $Y_j$ in the example above.

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Comparing the two test-score models

In Unit 4 we fit the test-score model without an interaction, $Y=\alpha_0+\alpha_{\text{hs}}X_{\text{hs}}+\alpha_{\text{iq}}X_{\text{iq}}+\epsilon$, and found $\hat\alpha_{\text{hs}}\approx5.95$ and $\hat\alpha_{\text{iq}}\approx0.56$. The model with the interaction gives $\hat\beta_{\text{hs}}\approx51.3$, $\hat\beta_{\text{iq}}\approx0.97$ and $\hat J_{\text{hs},\text{iq}}\approx-0.48$.

<ol type="a">
  <li>Write down the interpretation of $\alpha_{\text{iq}}$ and of $\beta_{\text{iq}}$. Which students does each one describe?</li>
  <li>Write down the interpretation of $\alpha_{\text{hs}}$ and of $\beta_{\text{hs}}$. Why is $\hat\beta_{\text{hs}}$ so much larger than $\hat\alpha_{\text{hs}}$?</li>
  <li>In the model with the interaction, what is the estimated difference between the two schooling groups for mothers with IQ $100$? How does it compare with $\hat\alpha_{\text{hs}}$?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Reading an interaction coefficient

A fitted model with two predictors is

$$ \hat y = 2 + 3X_1 + X_2 - 0.5\,X_1X_2. $$

<ol type="a">
  <li>What is the slope of $\hat y$ vs. $X_1$ when $X_2=0$? When $X_2=4$?</li>
  <li>At what value of $X_2$ is $X_1$ no longer associated with $Y$ at all?</li>
  <li>In words, what does $\hat\beta_1=3$ represent?</li>
  <li>Does $\hat\beta_1=3$ describe the effect of $X_1$ for a typical data point? Explain in one sentence.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Centering a predictor

In the test-score model we replaced $X_{\text{iq}}$ by $\Delta_{\text{iq}}=X_{\text{iq}}-\overline X_{\text{iq}}$ before forming the interaction.

<ol type="a">
  <li>Which coefficients does this change, and which does it leave alone?</li>
  <li>In the centered model $\hat\beta_{\text{hs}}\approx2.8$ and $\hat J_{\text{hs},\text{iq}}\approx-0.48$. What is the estimated difference between the two schooling groups for mothers with IQ $110$?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Choosing the horizontal axis

You fit a model with four predictors and want a residual plot.

<ol type="a">
  <li>What goes on the horizontal axis, and why not $Y_j$?</li>
  <li>Describe what the plot looks like when the model is adequate.</li>
  <li>You instead see a clear U-shape in the residuals. What does that suggest is missing from the model?</li>
</ol>
</div>

</details>

## 5.2 Feature maps for nonlinear models {#sec-5-2}

We now discuss how to build more complex models and directly assess their predictive power on out-of-sample data. With interactions, we already saw one way to extend a model: define a new predictor $X_3=X_1X_2$. More generally, defining a new predictor as *any* function of the existing ones lets us build very complex, flexible models that can still be analyzed within the linear regression framework. Let's formalize this for a single predictor.

The linear regression framework lets us fit models of the form

$$ Y\mid X \sim \text{Normal}(f(X),\ \sigma^2), $$

provided we can write $f(X)$ as a linear combination of nonlinear functions of $X$: functions $\phi_1(x),\dots,\phi_K(x)$ such that

$$ f(X) = \sum_{i=1}^K \beta_i\phi_i(X). $$

(An intercept is included by taking one of the features to be constant, $\phi_1(x)=1$.)

The requirement is that $f$ is **linear in the parameters**: a weighted sum of the features, with the unknown $\beta_i$ as the weights. The features $\phi_i$ must be *known* functions (no unknown parameters inside them), but they can be as nonlinear in $X$ as we like. So $\beta_0+\beta_1x+\beta_2x^2$ fits in this framework, even though it is not linear in $x$, while $\beta_0e^{\beta_1x}$ does not, because the parameter $\beta_1$ sits inside a nonlinear function. A common mistake is to think the model must be linear in $x$; it only has to be linear in the $\beta_i$.

The $\phi_i(X)$ are called <span class="term">[basis functions](https://en.wikipedia.org/wiki/Basis_function)</span>, or <span class="term">[features](https://en.wikipedia.org/wiki/Feature_(machine_learning))</span> in machine-learning terminology. We can think of each $\phi_i(X)$ as a new predictor. A common choice is $\phi_i(x)=x^i$, giving <span class="term">[polynomial regression](https://en.wikipedia.org/wiki/Polynomial_regression)</span>.

<div class="example" markdown="1">
#### Example (simulating a nonlinear model)

Consider $f(x)=\beta_0+\beta_1x+\beta_2x^2$ in the model above. This is a linear model once we define $X_1=\phi_1(X)=X$ and $X_2=\phi_2(X)=X^2$.

<u>Question:</u> (a) generate data from this model, and (b) fit it with `statsmodels`.

<u>Solution:</u> see the companion Colab notebook.
</div>

Which functions $\phi$ should we use? It depends on the problem &mdash; we might know something about the underlying physics of the data, or we might choose $\phi$ so the parameters have a clean interpretation (as with plain linear regression). The Mauna Loa CO$_2$ dataset from a previous class is one such example, where a linear trend plus a yearly cycle are natural features to include (see the companion Colab notebook).

Next, we'll see that the linear regression framework can fit very complicated nonlinear data &mdash; but we need to be careful when adding new features, since it's possible to "overshoot" and make the model too complex. That's the subject of the rest of this unit.

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Linear in the parameters?

Which of the following can be fit with the linear regression machinery of this unit (i.e. written as $\sum_i\beta_i\phi_i(x)$ for known $\phi_i$)? For those that can, give the features $\phi_i$. For those that can't, explain why.

<ol type="a">
  <li>$f(x)=\beta_0+\beta_1x+\beta_2x^3$</li>
  <li>$f(x)=\beta_0e^{\beta_1x}$</li>
  <li>$f(x)=\beta_0+\beta_1\ln x$</li>
  <li>$f(x)=\beta_0+\beta_1\sin(\beta_2x)$</li>
  <li>$f(x_1,x_2)=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_1x_2$</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Building the design matrix

Take features $\phi_1(x)=1$, $\phi_2(x)=x$, $\phi_3(x)=x^2$ and the three predictor values $x=0,1,2$.

<ol type="a">
  <li>Write out the $3\times3$ design matrix $A$ with $A_{k,i}=\phi_i(x_k)$.</li>
  <li>How many parameters does this model have, and why is fitting it to exactly three points a bad idea?</li>
</ol>
</div>

</details>

## 5.3 Cross-validation & the bias-variance tradeoff {#sec-5-3}

$R^2$ isn't very helpful for deciding how complex to make a model, since it's possible to explain *all* the variation in $Y$ with a model that's too complex. To address this, we use <span class="term">[cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))</span>: split the data into a <span class="term">[training set](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets)</span>, used to fit the model, and a <span class="term">[testing set](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets)</span>, used to evaluate the fitted model's predictions. The two sets must be **disjoint**: no data point is used both to fit the model and to test it, so the test error measures predictions on data the model has not seen.

Some notation: let $D=(X,Y)=\lbrace(X_1,Y_1),\dots,(X_N,Y_N)\rbrace$ denote the entire dataset, and $D^{\text{train}}=(X^{\text{train}},Y^{\text{train}})$, $D^{\text{test}}=(X^{\text{test}},Y^{\text{test}})$ the training/testing subsets, with $D=D^{\text{train}}\cup D^{\text{test}}$, $D^{\text{train}}\cap D^{\text{test}}=\emptyset$, and $N_{\text{train}},N_{\text{test}}$ points in each. Let $\hat y(x,D)=\sum_i\hat\beta_i\phi_i(x)$ denote the prediction of $E[Y\mid X{=}x]$ using coefficients fit on dataset $D$.

The <span class="term">[training error](https://en.wikipedia.org/wiki/Errors_and_residuals)</span> is

$$ \hat\sigma_{\text{train}}^2 = \frac{1}{N_{\text{train}}}\sum_{i=1}^{N_{\text{train}}} \big(\hat y(X_i^{\text{train}},D^{\text{train}}) - Y_i^{\text{train}}\big)^2, $$

which tells us how well the model predicts the same data used to fit it, and is closely related to $R^2$:

$$ R^2 \approx 1 - \frac{\hat\sigma_{\text{train}}^2}{\sigma_Y^2}. $$

Here $\sigma_Y^2=\text{var}(Y)$ is the variance of the response, estimated by the sample variance of the $Y_i$; recall from Unit 2 that $R^2$ is the fraction of this variance explained by the model.

To see how well the model predicts points it *wasn't* fit on, we use the <span class="term">[test error](https://en.wikipedia.org/wiki/Generalization_error)</span>,

$$ \hat\sigma_{\text{test}}^2 = \frac{1}{N_{\text{test}}}\sum_{i=1}^{N_{\text{test}}} \big(\hat y(X_i^{\text{test}},D^{\text{train}}) - Y_i^{\text{test}}\big)^2 $$

&mdash; note the coefficients used for $\hat y$ come from the *training* data, even though we evaluate at the test points.

<div class="example" markdown="1">
#### Example (cross-validation on a polynomial model)

Continuing the quadratic example above: plot the training and test error as a function of the number of parameters.

<u>Solution:</u> see the companion Colab notebook.
</div>

### Overfitting

To understand the U-shaped curve in the example above, we need some general terminology for estimators (of which $\hat y$ is an example). Recall from Unit 3 the <span class="term">[mean-squared error](https://en.wikipedia.org/wiki/Mean_squared_error)</span> of an estimator $\hat\theta$ of a quantity $\theta$,

$$ \text{MSE}_{\hat\theta} = E\big[(\hat\theta-\theta)^2\big], $$

and the bias-variance decomposition, $\text{MSE}_{\hat\theta} = \text{var}(\hat\theta) + \big(E[\hat\theta]-\theta\big)^2$.

Now return to a model $Y\mid X\sim\text{Normal}(f(X),\sigma^2)$ with $f(X)=\sum_i\beta_i\phi_i(X)$. Studying the MSE of our prediction of $f(x)$ given a dataset $D$ means setting $\hat\theta=\hat y(x,D)$, so

$$ \text{MSE}_{\hat y(x,D)} = E_D\big[(\hat y(x,D)-f(x))^2 \mid x\big], $$

where $E_D$ reminds us the expectation is over the distribution of the dataset $D$, with the predictor value $x$ held fixed. This is related to the test error $\hat\sigma_{\text{test}}^2$, but differs in two ways: (1) the test error averages over different $X$ points, and (2) it measures differences from the *observed* points, not the true function. For large enough $N_{\text{test}}$,

$$ \hat\sigma_{\text{test}}^2 \approx E_{X,Y}\big[(\hat y(X,D)-Y)^2 \mid D\big], \qquad Y = f(X)+\epsilon, $$

so the test error is the MSE for a fixed dataset, averaged over the predictor distribution (we write $D$ instead of $D^{\text{train}}$ for convenience, since they have the same distribution). It can be shown that

$$ E_{X,Y}\big[(\hat y(X,D)-Y)^2\mid D\big] = E_X\big[(\hat y(X,D)-f(X))^2\mid D\big] + \sigma_\epsilon^2, $$

where the second term doesn't depend on $X$, $D$, or the fitted model &mdash; only on the true data-generating distribution. Relating the first term to the MSE and taking expectations over $D$,

$$
\begin{aligned}
E_D[\hat\sigma_{\text{test}}^2] &= E_X\big[\text{MSE}_{\hat y(X,D)}\big] + \sigma_\epsilon^2\\
&= E_X\big[\text{var}(\hat y(X,D))\big] + E_X\big[(E_D[\hat y(X,D)]-f(X))^2\big] + \sigma_\epsilon^2.
\end{aligned}
$$

In practice we only observe a single dataset, but by resampling the training data we can effectively average over $D$ to approximate $E_D[\hat\sigma_{\text{test}}^2]$. So, up to a constant shift by $\sigma_\epsilon^2$, the test error obeys (on average) the same bias-variance decomposition as the MSE of any estimator &mdash; just with variance and bias terms that involve averages over both the data distribution and the predictor.

<div class="example" markdown="1">
#### Example (plotting bias and variance)

Continuing the polynomial cross-validation example: try to illustrate the bias-variance tradeoff by computing (approximations to) the bias and variance separately.

<u>Solution:</u> see the companion Colab notebook.
</div>

Summarizing what we've seen:

- As we add coefficients, a regression model becomes more flexible &mdash; it can fit more different datasets with a high $R^2$ / low training error. With enough features, a model can perfectly interpolate the data, $\hat y(X_i,D)=Y_i$ for every point.
- Unlike the training error, the test error $\hat\sigma_{\text{test}}^2$ doesn't simply keep decreasing as the model gets more complex &mdash; it has a **U-shape**. There's an optimal model size at which predictions on *new* data are best.
- A model that is too simple <span class="term">[underfits](https://en.wikipedia.org/wiki/Overfitting#Underfitting)</span>: it misses real structure, so both the training and the test error are high. A model that is too complex <span class="term">[overfits](https://en.wikipedia.org/wiki/Overfitting)</span>: it fits the noise in the training data, so its training error is low but its test error is high.
- The U-shape is explained by bias and variance: the MSE (the "math world" version of $\hat\sigma_{\text{test}}^2$) decomposes into a variance term (how much the model's predictions vary across different training sets) and a bias term (how far its predictions are, on average, from the truth).

The demo below draws two independent datasets from the *same* true function and fits a degree-$K$ polynomial to each. At low $K$ both fits miss the curve in the same way &mdash; high bias, low variance. Raise $K$ and they track their own data closely while disagreeing wildly with *each other* between the data points &mdash; low bias, high variance.

{% include_relative demos/bias_var.html %}

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Underfitting and overfitting

<ol type="a">
  <li>In your own words, what does it mean for a model to underfit? To overfit?</li>
  <li>For each, what do you expect the training error and the test error to look like?</li>
  <li>Which of the two goes with high bias, and which with high variance?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Reading a cross-validation table

A polynomial model of degree $K$ is fit to the same dataset for several $K$:

<table class="prob-table">
<tr><th>$K$</th><th>$\hat\sigma^2_{\text{train}}$</th><th>$\hat\sigma^2_{\text{test}}$</th></tr>
<tr><td>1</td><td>4.10</td><td>4.30</td></tr>
<tr><td>2</td><td>1.02</td><td>1.15</td></tr>
<tr><td>3</td><td>0.98</td><td>1.12</td></tr>
<tr><td>5</td><td>0.71</td><td>1.60</td></tr>
<tr><td>8</td><td>0.20</td><td>5.90</td></tr>
</table>

<ol type="a">
  <li>Which $K$ would you choose, and why?</li>
  <li>Why does $\hat\sigma^2_{\text{train}}$ never increase as $K$ grows?</li>
  <li>Which rows show underfitting, and which show overfitting?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Which term moves which way

As a model gains parameters, say whether each of the following typically increases, decreases, or stays about the same:

<ol type="a">
  <li>the variance term $E_X\big[\text{var}(\hat y(X,D))\big]$</li>
  <li>the squared-bias term $E_X\big[(E_D[\hat y(X,D)]-f(X))^2\big]$</li>
  <li>the noise term $\sigma_\epsilon^2$</li>
  <li>$R^2$ computed on the training data</li>
  <li>Use (a)&ndash;(c) to explain why the test error has a U-shape as the number of parameters grows.</li>
  <li>Why is the most complex model, which has the smallest training error, not the best choice for predicting new data?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### When the training and test sets overlap

Suppose we evaluate a polynomial model on a test set that, by mistake, contains some of the training points.

<ol type="a">
  <li>In the extreme case where the test set equals the training set, what is $\hat\sigma^2_{\text{test}}$?</li>
  <li>With partial overlap, is the test error too optimistic or too pessimistic? Why?</li>
  <li>How would the overlap change which number of parameters looks best?</li>
</ol>
</div>

</details>

## 5.4 Orthogonality & Fourier features {#sec-5-4}

Sometimes we have a specific hypothesis about which features contribute to a signal &mdash; e.g. for CO$_2$ data, we might suspect a linear trend plus a yearly cycle, since weather follows an annual pattern. But if we lack such prior knowledge, how should we choose the $\phi_i$? It's often advantageous to pick basis functions so the resulting predictors are *uncorrelated*: thinking of $\phi_i(X),\phi_j(X)$ as random variables (randomness coming from $X$), we'd like them uncorrelated with each other.

Assume $E[\phi_i(X)]=0$ throughout (easy to arrange by subtracting means). We'd like

$$ \text{cov}(\phi_i(X),\phi_j(X)) = 0, \qquad i \ne j. $$

When this holds for some distribution of $X$, we say $\phi_i,\phi_j$ are <span class="term">[orthogonal](https://en.wikipedia.org/wiki/Orthogonal_functions)</span> with respect to that distribution &mdash; "orthogonal" here just means uncorrelated.

<div class="example" markdown="1">
#### Example (sine series)

Let $X\sim\text{Uniform}(-L,L)$ (even non-random, evenly-spaced $X$ points behave statistically like a uniform random sample, so this approximates evenly-spaced data too). Consider the basis functions $\phi_j(x)=\sin(\pi xj/L)$.

<u>Question:</u> show via simulation that the $\phi_j$ are orthogonal with respect to the uniform distribution on $[-L,L]$.

<u>Solution:</u> see the companion Colab notebook.
</div>

If the $\phi_j$ are orthogonal, the regression coefficients simplify to

$$ \beta_j = \frac{\text{cov}(Y,\phi_j(X))}{\text{var}(\phi_j(X))} = \frac{E[Y\phi_j(X)]}{E[\phi_j(X)^2]}, $$

so the fitted coefficients from $N$ data points are approximately

$$ \hat\beta_j \approx \frac{\sum_{i=1}^N \phi_j(X_i)Y_i}{\sum_{i=1}^N \phi_j(X_i)^2}. $$

This means $\hat\beta_j$ should depend only weakly on how many other features we've included &mdash; a consequence of orthogonality. (Contrast this with the earlier polynomial-feature example, where adding a new predictor like $X^3$ could dramatically change the fitted $\hat\beta_2$.)

A limitation of the sine-only model: it can only represent "odd" functions ($f(-x)=-f(x)$), since every $\phi_j$ has that property. To fix this while staying orthogonal, add cosine features $\cos(\pi xj/L)$. The resulting model, called a <span class="term">[Fourier series](https://en.wikipedia.org/wiki/Fourier_series)</span>,

$$ f(x) = \beta_0 + \sum_{i=1}^K \Big[\beta_i\sin\Big(\frac{\pi ix}{L}\Big) + \alpha_i\cos\Big(\frac{\pi ix}{L}\Big)\Big], $$

(writing $\alpha_i$ for the cosine coefficients) is one of the most important models in data science and engineering &mdash; as $K\to\infty$, it can be shown to approximate essentially *any* function.

<div class="example" markdown="1">
#### Example (Fourier series in Python)

Suppose data come from $f(x)=\sin(3\pi x)+\sin(10\pi x)$ with $X_i=(i-1)/N$ (evenly spaced on $[0,1]$).

<u>Question:</u> for different values of $\sigma^2$, generate data from this model and fit a Fourier series with $K=100$ terms.

<u>Solution:</u> see the companion Colab notebook.
</div>

Computing the coefficients $\hat\beta_j,\hat\alpha_j$ for a Fourier series is called a <span class="term">[discrete Fourier transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)</span> (DFT) &mdash; usually reserved for the case of equally-spaced $X_i$. In that case, the orthogonality condition holds exactly in "data world," not just "math world." Precisely, for predictor values $X_i = L(i-1)/(N-1)$, $i=1,\dots,N$ on $[0,L]$ (so $X_1=0,\dots,X_N=L$; generate these with `np.linspace`):

**Theorem.** $\displaystyle\sum_{i=1}^N \sin\Big(\frac{2\pi jX_i}{L}\Big)\sin\Big(\frac{2\pi kX_i}{L}\Big) = 0$ for $j\ne k$ with $1\le j,k<(N-1)/2$.

The restriction matters: the sum is over $N$ points whose spacing makes the sines periodic with period $N-1$, so orthogonality fails once $j+k$ reaches $N-1$ (try $N=12$ with $j=5,k=6$). This is the discrete analogue of *aliasing* &mdash; frequencies above half the sampling rate can't be distinguished from lower ones.

Often we want to summarize how different frequencies are represented in our data without caring whether they come from the sine or cosine terms (note the sines in the theorem differ slightly from the earlier example, since the interval changed from $[-L,L]$ to $[0,L]$). For this we use the <span class="term">[power spectral density](https://en.wikipedia.org/wiki/Spectral_density)</span>, also called the <span class="term">[periodogram](https://en.wikipedia.org/wiki/Periodogram)</span>,

$$ P_j = \hat\beta_j^2+\hat\alpha_j^2, $$

a fundamental object in signal processing that tells us how "wobbly" a signal is at each frequency.

<div class="example" markdown="1">
#### Example (periodogram)

<u>Question:</u> compute the periodogram of the data from the Fourier-series example above, and confirm it matches the output of `scipy.signal.periodogram`. (A neat, underappreciated fact: this fundamental signal-processing tool comes directly from fitting a linear regression model with least squares!)

<u>Solution:</u> see the companion Colab notebook.
</div>

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Orthogonality depends on the distribution

Take the two features $\phi_1(x)=x$ and $\phi_2(x)=x^2$.

<ol type="a">
  <li>Compute $\text{cov}(\phi_1(X),\phi_2(X))$ when $X\sim\text{Uniform}(-1,1)$.</li>
  <li>Compute it again when $X\sim\text{Uniform}(0,1)$.</li>
  <li>In which case are the features orthogonal, and why does the answer depend on the distribution of $X$ rather than on the $\phi_i$ alone?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Orthogonal features and fitted coefficients

Suppose centered predictors $X_1$ and $X_2$ are orthogonal in the observed data, so $\sum_{i=1}^N X_{1i}X_{2i}=0$. Fit the no-intercept model $Y=\beta_1X_1+\beta_2X_2+\epsilon$.

<ol type="a">
  <li>Show that $\hat\beta_1=\sum_iX_{1i}Y_i/\sum_iX_{1i}^2$ and write the analogous formula for $\hat\beta_2$.</li>
  <li>If $X_2$ is removed from the model, does the fitted coefficient of $X_1$ change? Explain.</li>
  <li>What part of this argument fails when $\sum_iX_{1i}X_{2i}\ne0$?</li>
</ol>
</div>

</details>

## Problems {#problems}

<div class="exercise" markdown="1">
#### Problem 5.1 &mdash; Car brands and mpg

Consider a dataset of cars and their miles per gallon:

```python
data = pd.read_csv("https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Auto.csv", encoding="ISO-8859-1")
data["name"] = [name.split()[0] for name in data["name"].values]
```

The second line extracts the brand (e.g. Toyota) from the full model name. We'll study which brands have the best mpg. Some brands tend to make larger, heavier cars (e.g. pickup trucks) with worse mpg, but we want to compare brands *within* a given type of car &mdash; controlling for factors like year and weight.

<ol type="a">
<li>Using all columns <strong>except</strong> <code>origin</code> and <code>displacement</code> (its units aren't clear), write down the regression model you'd fit to address this question, assuming no interactions. Interpret each coefficient.</li>
<li>Fit the model.</li>
<li>What are the $5$ best brands for mpg, within the same type of car?</li>
<li>Try improving the model by adding interaction terms. Do you find any statistically significant interactions?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 5.2 &mdash; Marginal regression in an interactions model

Consider

$$
\begin{aligned}
&X_1 \sim \text{Normal}(0,\sigma_1^2), \qquad X_2 \sim \text{Normal}(0,\sigma_2^2),\\
&Y\mid(X_1,X_2) \sim \text{Normal}(\beta_1X_1+\beta_2X_2+J_{1,2}X_1X_2,\ \sigma^2).
\end{aligned}
$$

<ol type="a">
<li>Derive the distributions of $Y\mid X_1$ and $Y\mid X_2$. (Hint: both are Normal, so you just need the mean and variance.)</li>
<li>When does this probability model define a valid single-predictor regression model for $Y$ vs. $X_i$ ($i=1,2$) if we ignore the other predictor? Would this still hold if the predictors didn't have zero mean?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 5.3 &mdash; Predicting the residual plot from an interaction model

Suppose we have $200$ data points from $Y = 4X_1-2X_2+4X_1X_2+\epsilon$, with $\sigma=0.2$, $X_1$ uniform on $[-1,1]$, and $X_2$ a Bernoulli predictor (assume $X_2=0$ for about half the points). The goal is to build intuition about residual plots.

<ol type="a">
<li><strong>Without fitting a regression</strong>, describe in detail what the residual plot would look like if we fit a linear model with <em>no</em> interaction term. First think about what the data looks like when $X_2=0$ and $X_2=1$ separately &mdash; sketch the regression line for each and note the variation around it. Then think about what the (no-interaction) fitted line looks like given this picture: what's a rough estimate of the fitted slopes $\hat\beta_1,\hat\beta_2$? Finally, to get a sense of the residuals, take the difference between the true model and this line.</li>
<li>Confirm your answer with simulation.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 5.4 &mdash; Drug interactions

When treating microbial infections or cancer, drug combinations can outperform individual drugs, but identifying the best combination is hard because high-order interactions are difficult to pin down. To study how best to combine $M$ drugs, consider a regression where $Y$ is the drug "effect" and $X_i$ is a Bernoulli variable for whether drug $i$ is present:

$$
\begin{aligned}
Y &= \sum_{i=1}^M \beta_iX_i + \sum_{i=1}^M\sum_{j>i}^M J_{i,j}X_iX_j + \sum_{i=1}^M\sum_{j>i}^M\sum_{k>j}^M J_{i,j,k}X_iX_jX_k\\
&\quad + \cdots + J_{1,2,\dots,M}X_1X_2\cdots X_M + \epsilon.
\end{aligned}
$$

For example, with $M=3$:

$$
\begin{aligned}
Y &= \beta_1X_1+\beta_2X_2+\beta_3X_3+J_{1,2}X_1X_2+J_{1,3}X_1X_3\\
&\quad +J_{2,3}X_2X_3+J_{1,2,3}X_1X_2X_3+\epsilon.
\end{aligned}
$$

<ol type="a">
<li>Suppose $M=3$ with
$$ \begin{bmatrix}\beta_1\\\beta_2\\\beta_3\\J_{1,2}\\J_{1,3}\\J_{2,3}\\J_{1,2,3}\end{bmatrix} = \begin{bmatrix}1.2\\-0.8\\-0.11\\3.48\\-2.62\\1.03\\1.66\end{bmatrix}. $$
Interpret each coefficient.</li>
<li>What's the optimal treatment &mdash; which combination of drugs $1,2,3$ maximizes $Y$? (One approach: enumerate every $(X_1,X_2,X_3)$, compute $Y$ for each, and find the maximizer with a loop or <code>argmax</code>.)</li>
<li>Now suppose $\sigma^2=1$. By simulating $Y$ under these parameters for different $N$, determine how many data points are needed to reliably find all interaction coefficients statistically significant ($p<0.05$).</li>
<li>Repeat (c) but fit a model with no interactions. What do you find? How does including the interaction terms change the $p$-values?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 5.5 &mdash; Interaction term correlations

Consider two binary predictors $X_1,X_2\in\lbrace0,1\rbrace$ with joint distribution

<table class="prob-table">
<tr><th>$P(X_1,X_2)$</th><th>$X_2=0$</th><th>$X_2=1$</th></tr>
<tr><th>$X_1=0$</th><td>0.2</td><td>0.3</td></tr>
<tr><th>$X_1=1$</th><td>0.1</td><td>0.4</td></tr>
</table>

<ol type="a">
<li>Are $X_1$ and $X_2$ independent?</li>
<li>Adding an interaction term $X_3=X_1X_2$, find the correlation coefficients between $X_3$ and $X_1$, and between $X_3$ and $X_2$.</li>
<li>Now generate data from a model with the interaction term, and fit a regression model with the interaction term. Is the correlation between $\hat\beta_1$ and $\hat J_{1,2}$ consistent with your calculation in part (b)?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 5.6 &mdash; Mean-squared error

Prove the formula

$$ E_{X,Y}\big[(\hat y(X,D)-Y)^2\mid D\big] = E_X\big[(\hat y(X,D)-f(X))^2\mid D\big] + \sigma_\epsilon^2. $$
</div>

<div class="exercise" markdown="1">
#### Problem 5.7 &mdash; Impulse (indicator) features

Consider $Y\mid X \sim \text{Normal}\big(\sum_{i=1}^K\beta_i\phi_i(X),\ \sigma^2\big)$, with $X\in[0,1)$. Define the intervals $I_i=[(i-1)/K,\ i/K)$, so $[0,1) = I_1\cup I_2\cup\cdots\cup I_K$ (each $x\in[0,1)$ lies in exactly one $I_i$), and features

$$ \phi_i(x) = \begin{cases} 1 & x \in [(i-1)/K,\ i/K) \\ 0 & x \notin [(i-1)/K,\ i/K). \end{cases} $$

<ol type="a">
<li>Are the $\phi_i$ orthogonal with respect to a random variable $X$ on $[0,1)$? Does the answer depend on the distribution of $X$?</li>
<li>Using <code>statsmodels</code>, fit a model with these features (make up your own simulated dataset, or reuse the Fourier/polynomial-model code from class). Write a function <code>phi(x, i)</code> that takes the predictor array and returns $[\phi_i(X_1),\dots,\phi_i(X_N)]$. Use $K=10$, $N=100$.</li>
<li>Let $\hat\beta_j$ be the least-squares fitted value of $\beta_j$. Show that in this model, $\hat\beta_j$ is simply the average of $Y_i$ among points with $X_i\in I_j$:
$$ \hat\beta_j = \frac{1}{N_j}\sum_{i:\,X_i\in I_j} Y_i, $$
where $N_j$ is the number of points in $I_j$.</li>
</ol>
</div>
