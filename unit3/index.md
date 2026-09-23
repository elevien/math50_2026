---
layout: notes
title: "Unit 3: Estimators, and uncertainty Quantification for single-Predictor Regression"
unit_title: "Unit 3"
unit_subtitle: "The CLT, uncertainty quantification, and single-predictor regression"
toc:
  - {href: "#sec-3-1", label: "3.1 The LLN & the CLT"}
  - {href: "#sec-3-2", label: "3.2 Estimators, bias & consistency"}
  - {href: "#sec-3-3", label: "3.3 Confidence intervals"}
  - {href: "#sec-3-4", label: "3.4 Least squares and estimators for linear regression"}
  - {href: "#sec-3-5", label: "3.5 Autoregressive models and the Hurwicz bias"}
  - {href: "#sec-3-6", label: "3.6 Hypothesis testing"}
  - {href: "#problems", label: "Problems"}
---

<div class="unit-overview" markdown="1">

In this unit, we address the question of statistical error. We begin with the **Law of Large numbers** and **Central Limit Theorem**, which connects the behavior of large random samples to the Normal distribution from Unit 2. We then define an **estimator** (somthing we hav already examples of) and discuss the concepts of **bias**, **variance**,  **confidence intervals** and the sample distribution. We apply these ideas to **least squares estimator** for the regression coefficient in a linear regression model and autoregressive models. Finally we discuss **hypothesis testing** briefly. 

#### Concepts

The law of large numbers, the Central Limit Theorem, estimators, sample distributions, bias and consistency, standard errors, confidence intervals, the bias-variance decomposition, fitting the single-predictor linear regression model via least squares, autoregressive models and the Hurwicz bias, and hypothesis testing and $p$-values for regression models.

#### Things to practice

- Use the CLT to approximate the distribution of a sum or sample average, and to justify treating an estimator's sample distribution as approximately Normal.
- Identify whether a simple estimator is biased or not, by hand if possible, or using simulations in Python.
- Know the properties of least square estimator and what the sample distribution looks like. 
- Perform a hypothesis test by simulating the model. 


</div>

<p class="pdf-link"><a href="unit3.pdf">Unit 3 notes pdf</a> <a href="https://colab.research.google.com/github/elevien/math50_2026/blob/main/unit3/unit3.ipynb">Unit 3 notebook</a></p>

## 3.1 The Law of Large Numbers & the Central Limit Theorem {#sec-3-1}

The goal of this section is to understand the distribution of a sum of iid random variables when $N$ is large &mdash; this matters if we want to be precise about how accurate our estimates are, which is the whole subject of this unit. We begin with the <span class="term">[law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers)</span>, which makes precise the statement that a sample average approximates the expectation.

### The Law of Large Numbers

The binomial distribution illustrates a basic principle we've already used a number of times: when we sum many independent random variables and divide by the total number, the result lands close to the mean.

**Theorem (Law of Large Numbers).** Let $X_i$ be iid and $S_N = \sum_{i=1}^N X_i$. If $E[X_i]<\infty$, then $S_N/N \to E[X_i]$.

This isn't fully precise &mdash; we'd need to define what it means for a random quantity to "converge" &mdash; but for our purposes it suffices to think of this as: for large enough $N$, $S_N/N$ won't differ from $E[X_i]$ by much. Another way to say it: for iid $X_i$, the sample average $\overline X$ approaches $E[X_i]$.

The binomial distribution tells us more: the variation around $E[X_i]$ is proportional to $1/\sqrt N$. It's natural to ask whether this holds for other random variables too. The key insight is that the $N$-dependence in $E[Y]=qN,\ \operatorname{var}(Y)=Nq(1-q)$ doesn't actually depend on the underlying distribution being Bernoulli &mdash; so if $X_i$ is a die roll, or a uniform random variable, we'd expect the same scaling to hold. The behavior of random sums is, in fact, even more universal than this: we can describe the distribution of *any* random sum (with finite variance) using a single distribution &mdash; the Normal distribution from Unit 2.

### The Central Limit Theorem and sample distribution

We can now state the <span class="term">[Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)</span> (CLT) precisely.

**Theorem.** Let $X_i$ be iid with $E[X_i]=\mu$, $\operatorname{var}(X_i)=\sigma^2$, and let $S_N=\sum_{i=1}^N X_i$. Then

$$ P\left(\frac{S_N-N\mu}{\sqrt{N\sigma^2}}<z\right) \to P(Z<z), \qquad Z \sim \text{Normal}(0,1). $$

The variable $Z$ is called a <span class="term">[standard normal](https://en.wikipedia.org/wiki/Normal_distribution#Standard_normal_distribution)</span> random variable. Its CDF is common enough that we abbreviate $\Phi(z)=P(Z<z)$, and write $\phi(z)$ for its pdf. Using the properties of Normal random variables from Unit 2, we can restate the CLT informally as $S_N \approx \text{Normal}(N\mu,\ N\sigma^2)$. In the demo below, the histogram is of standardized sums $(S_N-N\mu)/\sqrt{N\sigma^2}$ and the curve is the standard Normal density: whatever distribution you draw the $X_i$ from, the histogram closes on that curve as $N$ grows.


**Note on the iid assumption:** the version of the CLT stated above assumes the $X_i$ are independent and identically distributed. There are more advanced versions that allow some dependence, but not arbitrary dependence; strong correlation can change the sample distribution substantially. For this course, check first whether iid is a reasonable approximation before using the CLT.

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Standardizing a sum

Let $X_1,\dots,X_N$ be iid with $E[X_i]=\mu$ and $\operatorname{var}(X_i)=\sigma^2$, and let $S_N=\sum_i X_i$.

<ol type="a">
  <li>Find $E[S_N]$ and $\operatorname{var}(S_N)$.</li>
  <li>Write the standardized version of $S_N$.</li>
  <li>Use the CLT to write an approximate distribution for $S_N$.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Binomial approximation

Let $Y = \sum_{i=1}^N X_i$ where $X_1,\dots,X_N$ are iid $\text{Bernoulli}(q)$ (this is what it means to say $Y\sim\operatorname{Binomial}(N,q)$).

<ol type="a">
  <li>Use the CLT to approximate the distribution of $Y$.</li>
  <li>Use that approximation to estimate $P(Y<N/2)$ when $N=200$ and $q=0.3$.</li>
  <li>Write Python code using <code>scipy.stats.norm</code> and <code>scipy.stats.binom</code> to compare the CLT approximation with the exact Binomial probability.</li>
  <li>Explain why the approximation should improve as $N$ increases.</li>
</ol>
</div>

</details>

## 3.2 Estimators, bias & consistency {#sec-3-2}

We've already aluded to concept of statistical inference and parameter estimation.  Recall that $\hat q = \overline Y$ is an estimator of the parameter $q$ for a Bernoulli distribution, and more generally we've gone back and forth between "data world" (sample means, histograms) and "math world" (expectation, density) via sample averages.

**Statistical inference is the process of estimating the parameters of a distribution (e.g. $\mu$ and $\sigma$) from samples of $Y$, *and* expressing our uncertainty in these estimates.** The "expressing uncertainty" part is what we haven't yet discussed formally.

In general, an <span class="term">[estimator](https://en.wikipedia.org/wiki/Estimator)</span> $\hat\theta$ of a parameter $\theta$ is anything we compute from the data that's meant to approximate $\theta$ &mdash; in principle this could be any quantity (like the maximum value), but it will almost always take the form of a sample average in some way, as we'll see. The key point: **the estimator is a function of the data** &mdash; $\hat\theta$ depends on the specific sample or simulation we run, whereas the parameter it approximates does not depend on the data and is (in classical statistics) a fixed number. For instance, in a YES/NO survey or a two-candidate election, the "true" quantity we want to measure is the population fraction who would answer YES. Our estimate $\hat q$ is a random variable that depends on which subset of the population we happen to sample, and it changes if we sample a different subset.

### Sample distribution and standard errors

We call the distribution of $\hat\theta$ over many replications of our data the <span class="term">[sample distribution](https://en.wikipedia.org/wiki/Sampling_distribution)</span>. We use <span class="term">[replicate](https://en.wikipedia.org/wiki/Replication_(statistics))</span> to mean a different realization of the *entire* dataset (as opposed to the individual samples *within* a dataset) &mdash; the terminology is a little confusing: the sample distribution is the distribution of $\hat\theta$ over many replicates, but each replicate itself consists of many samples. The demo below makes the distinction concrete: the pale bars are a histogram of the $N$ *samples* $Y_i$ inside the latest replicate, while the solid bars collect one estimate $\hat\mu=\overline Y$ per replicate &mdash; that second histogram is the sample distribution, compared against the theoretical $\text{Normal}(\mu,\sigma^2/N)$ curve.

{% include_relative demos/replicates.html %}

<div class="example" markdown="1">
#### Example (sample distribution of a Normal mean)

Suppose $Y \sim \text{Normal}(\mu,\sigma^2)$.

<u>Question:</u> what is the sample distribution of $\hat\mu$ (our estimator of $\mu$)?

<u>Solution:</u>

$$ \hat\mu = \overline Y = \frac{1}{N}\sum_{i=1}^N Y_i. $$

Because sums of independent Normal random variables are exactly Normal,

$$ \sum_{i=1}^N Y_i \sim \text{Normal}(N\mu,\ N\sigma^2). $$

For non-Normal variables with finite variance, the CLT gives this same statement as a large-sample approximation. Dividing by $N$,

$$ \hat\mu \sim \text{Normal}\Big(\mu,\ \frac{\sigma^2}{N}\Big). $$

This assumes $\sigma$ is known.
</div>

A natural way to quantify our uncertainty in $\hat\mu$ is the standard deviation of $\hat\mu$ under the sample distribution. We call this the <span class="term">[standard error](https://en.wikipedia.org/wiki/Standard_error)</span>: our estimate of the standard deviation of the sample distribution. For the Normal model, if we're estimating the mean and happen to know $\sigma$,

$$ \text{se}(\hat\mu) = \frac{\sigma}{\sqrt N}. $$

This tells us how much our estimate would vary between different experiments (or surveys, or simulations). Importantly, the standard error depends on $\sigma$, which we usually *don't* know! So it's common to plug in an estimate $\hat\sigma$ instead:

$$ \text{se}(\hat\mu) = \frac{\hat\sigma}{\sqrt N}. $$

It should be clear from context which one is meant: if we're working with real data and don't know $\sigma$, "standard error" means the second formula; if we're working with a model where the parameters are specified, it means the first.

### Bias and consistency

There must be some minimal properties we'd like an estimator to have. At the very least, it should be informed by the data in the sense that more data brings $\hat\theta$ closer to the true $\theta$ &mdash; more precisely, larger $N$ should make $\hat\theta$ closer to $\theta$. Of course we need to say what "closer" means for random quantities. We'll say $\hat\theta$ is <span class="term">[consistent](https://en.wikipedia.org/wiki/Consistent_estimator)</span> if

$$ E[\hat\theta] \to \theta \quad\text{and}\quad \text{se}(\hat\theta) \to 0 \quad\text{as } N\to\infty, $$

i.e. as we collect more data, the sample distribution becomes more concentrated around $\theta$.

Consistency alone isn't the whole story: $\hat\mu_1 = \hat\mu + 1/N$ is also consistent, yet clearly worse than $\hat\mu$. To rule this out we say $\hat\theta$ is <span class="term">[unbiased](https://en.wikipedia.org/wiki/Bias_of_an_estimator)</span> if, for *every* $N$ (not just large $N$), the average over the sample distribution equals the true value:

$$ E[\hat\theta] = \theta. $$

<div class="example" markdown="1">
#### Example (bias and consistency)

For a Normal random variable, define the estimator of the mean

$$ \hat\mu_2 = \frac{Y_1+Y_2}{2}. $$

<u>Question:</u> is $\hat\mu_2$ biased? Is it consistent? What's its sample distribution?

<u>Solution:</u> $\hat\mu_2$ has sample distribution

$$ \hat\mu_2 \sim \text{Normal}\Big(\mu,\ \frac{\sigma^2}{2}\Big). $$

It is therefore unbiased, but *not* consistent, since $\text{se}(\hat\mu_2) = \sigma/\sqrt2$ doesn't shrink as we collect more data (it only ever uses two samples).
</div>

<div class="example" markdown="1">
#### Example (Bessel's correction)

Now consider estimating the variance of $Y \sim \text{Normal}(\mu,\sigma^2)$. Given samples $Y_1,\dots,Y_n$, the natural way to estimate $\sigma^2$ is

$$ \text{var}(Y) = E\big[(Y-E[Y])^2\big] \approx \hat\sigma_0^2 := \frac1n\sum_{i=1}^n (Y_i-\overline Y)^2. $$

It turns out $\hat\sigma_0^2$ is *biased* &mdash; in fact

$$ \hat\sigma^2 := \frac{1}{n-1}\sum_{i=1}^n (Y_i-\overline Y)^2 = \frac{n}{n-1}\hat\sigma_0^2 $$

is unbiased. The correction factor $n/(n-1)$ is called <span class="term">[Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)</span>.

<u>Question:</u> confirm with simulated data that $\hat\sigma_0^2$ is biased while $\hat\sigma^2$ is not.
</div>

### Bias-variance tradeoff

Define the <span class="term">[mean-squared error](https://en.wikipedia.org/wiki/Mean_squared_error)</span> of an estimator $\hat\theta$ of a quantity $\theta$ (which could be a parameter, or the value of a function like $f(x)$ we're trying to predict):

$$ \text{MSE}_{\hat\theta} = E\big[(\hat\theta-\theta)^2\big]. $$

The following is the key result behind the <span class="term">[bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)</span> &mdash; the U-shaped tradeoff curves we'll see later in the course.

**Theorem (bias-variance decomposition).**

$$ \text{MSE}_{\hat\theta} = \text{var}(\hat\theta) + \big(E[\hat\theta]-\theta\big)^2. $$

<details markdown="1">
<summary>Proof</summary>

Using the definition of variance,

$$ \text{var}(\hat\theta-\theta) = E\big[(\hat\theta-\theta)^2\big] - \big(E[\hat\theta-\theta]\big)^2. $$

Since $\theta$ is a constant, $\text{var}(\hat\theta-\theta)=\text{var}(\hat\theta)$; rearranging gives the result.
</details>

An estimator's error therefore decomposes into how *variable* it is (variance) plus how far off it is *on average* (squared bias).

<details class="optional-section" open markdown="1">
<summary><strong>Aside: maximum likelihood estimation</strong> <span class="optional-badge">(optional)</span></summary>

Sometimes the "right" estimator for a parameter is obvious &mdash; that's the case for $q$ in the Bernoulli distribution. But this won't always be true, so it's useful to have a more systematic way of finding estimators.

Recall the Binomial pmf,

$$ P(Y) = \binom{n}{Y} q^Y (1-q)^{n-Y}. $$

In statistics we sometimes call this the <span class="term">[likelihood](https://en.wikipedia.org/wiki/Likelihood_function)</span>, written $L(Y\mid q)$ &mdash; the notation suggests we're thinking of $P$ as a distribution conditioned on a particular parameter value. More generally, the likelihood is the probability of the observed data given the parameters; this notation foreshadows Bayesian thinking, where the parameters themselves are treated as random variables (more on that later).

The equation above tells us how likely it is to observe $Y$ YESes among $n$ people surveyed. It seems reasonable that this shouldn't be too small, since that would mean our results are an anomaly &mdash; the larger $L(Y\mid q)$ is, the more "likely" our data. This suggests estimating $q$ by the value $\hat q$ that makes $L(Y\mid q)$ largest: the <span class="term">[maximum likelihood estimate](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)</span> (MLE). Using calculus (try it!), the maximizer is

$$ \hat q_{\text{MLE}} = \frac{Y}{n}. $$

For a Normal distribution with mean $\mu$ and variance $\sigma^2$, the MLEs are the usual sample mean and (biased) sample variance we've already seen.

</details>

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Bias and consistency

Let $X \sim \operatorname{Bernoulli}(q)$ and $X_1,\dots,X_N$ denote $N$ samples of $X$. For each estimator of $q$, write down the standard error and state whether it is unbiased and/or consistent. You can use exact variance formulas; you do not need the CLT.

<ol type="a">
<li>
$$ \hat q_0 = \frac1N\sum_{i=1}^N X_i $$
</li>
<li>
$$ \hat q_1 = \frac{Y}{N} + \frac{1}{\sqrt N}, \qquad Y=\sum_{i=1}^N X_i $$
</li>
<li>
$$ \hat q_2 = \frac{1}{\lfloor N/2\rfloor}\sum_{i=1}^{\lfloor N/2\rfloor} X_i $$
The notation $\lfloor n\rfloor$ means the largest integer $\le n$. For example, $\lfloor 101/2\rfloor=50$.
</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Comparing two estimators

Let $X_1,\dots,X_N$ be iid with $E[X_i]=\mu$ and $\operatorname{var}(X_i)=\sigma^2$, and consider

$$ \hat\mu_A=\overline X, \qquad \hat\mu_B=\frac{1}{N}\sum_{i=1}^N X_i+\frac{2}{N}. $$

<ol type="a">
  <li>Find the bias and standard error of each estimator.</li>
  <li>Which estimators are unbiased? Which are consistent?</li>
  <li>Compare their mean-squared errors. Which estimator would you prefer for finite $N$?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Bessel's correction

For $n=2$ iid samples $Y_1,Y_2$ with mean $\mu$ and variance $\sigma^2$,

$$ \hat\sigma_0^2 = \frac12\big[(Y_1-\overline Y)^2+(Y_2-\overline Y)^2\big], \qquad \overline Y = \frac{Y_1+Y_2}{2}. $$

<ol type="a">
<li>Show that $\hat\sigma_0^2 = \frac14(Y_1-Y_2)^2$.</li>
<li>Using $\text{var}(Y_1-Y_2)=2\sigma^2$, compute $E[\hat\sigma_0^2]$ in terms of $\sigma^2$.</li>
<li>Confirm this matches $\frac{n-1}{n}\sigma^2$ for $n=2$.</li>
</ol>
</div>

</details>

## 3.3 Confidence intervals {#sec-3-3}

The idea of a <span class="term">[confidence interval](https://en.wikipedia.org/wiki/Confidence_interval)</span> is, roughly, to describe the range of values where the true $\theta$ plausibly lies, given an estimate $\hat\theta$ and its sample distribution. We'll mostly use the $95\%$ confidence interval (95%-CI),

$$ \Big[\hat\theta - 1.96\,\text{se}(\hat\theta),\ \ \hat\theta + 1.96\,\text{se}(\hat\theta)\Big]. $$

For an approximately Normal sample distribution, $95\%$ of values lie within $1.96$ standard errors of its center; this is why the resulting intervals have about $95\%$ coverage.

Note that samples from *this* interval do not have the same distribution as $\hat\theta$ over replicates of our data. Put another way: if we draw many samples from our estimated sample distribution, their spread is not the same as the spread of $\hat\theta$ we'd get by actually running the experiment many times. The correct interpretation of the 95%-CI is: **if we generate many replicates of the data, the true value $\theta$ will fall inside the CI for $95\%$ of them.**

Technically, it is *not* true that there's a $95\%$ chance the true value lies in the particular 95%-CI computed from one dataset. To see why, note that $\theta$ has a $95\%$ chance of lying in

$$ \Big[\theta - 1.96\,\text{std}(\hat\theta),\ \ \theta + 1.96\,\text{std}(\hat\theta)\Big], $$

but this is different from the interval above, since here we've replaced $\hat\theta$ with $\theta$. The distinction is illustrated in the demo below, where each horizontal bar is the 95%-CI from one replicate and is coloured by whether it contains the true value &mdash; every CI is centered at a different (random) $\hat\theta$, while the dashed line marking $\theta$ stays fixed. It's a subtle point, and for practical purposes you can mostly think of the 95%-CI as "the region where the parameter is likely to be." (We'll revisit this with a genuinely different, Bayesian, interpretation later in the course.)

{% include_relative demos/notes3CI.html %}

<div class="example" markdown="1">
#### Example (sample size for a target CI width)

Suppose we're designing an experiment. Our model is Normal and, from previous experience, we have a ballpark estimate of the standard deviation, $\sigma=1$.

<u>Question:</u> roughly how many samples do we need for the $95\%$ confidence interval to have margin of error less than $1$?

<u>Solution:</u> the standard error based on $n$ samples is $\sigma/\sqrt n = 1/\sqrt n$, so the CI is

$$ \Big[\hat\mu - \frac{1.96}{\sqrt n},\ \ \hat\mu + \frac{1.96}{\sqrt n}\Big], $$

with width $2\times 1.96/\sqrt n = 3.92/\sqrt n$. This interval contains the true value for $95\%$ of replicates, so margin of error less than $1$ means width $<2$:

$$ \frac{3.92}{\sqrt n} < 2 \implies \sqrt n > 1.96 \implies n > (1.96)^2 \approx 3.84. $$

We can check this by running many replicates for each $n$ (as in the class notebook).
</div>

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Reading a confidence interval

Suppose an estimator has $\hat\theta=4.8$ and $\operatorname{se}(\hat\theta)=0.7$.

<ol type="a">
  <li>Compute the approximate $95\%$ confidence interval.</li>
  <li>Explain what "$95\%$ confidence" means in terms of repeated datasets.</li>
  <li>Explain why it is not literally a statement that $\theta$ has probability $0.95$ of lying in this particular interval.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Target width

For a Normal model with known $\sigma=2$, find the minimum $n$ so that the approximate $95\%$ confidence interval for $\mu$ has total width at most $1$.
</div>

</details>

## 3.4 Least squares and estimators for linear regression {#sec-3-4}

We now discuss the estimator for the regression coefficient in a single predictor linear regression model. Recall from Unit 2 that a <span class="term">[linear regression](https://en.wikipedia.org/wiki/Linear_regression) model</span> is

$$ Y = \beta_0+\beta_1X+\epsilon,\qquad \epsilon\sim\text{Normal}(0,\sigma^2), $$

The input is the <span class="term">[predictor](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)</span> ($X$) and the output is the <span class="term">[response variable](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)</span> ($Y$). 

Also recall from [Unit 2](../unit2/#sec-2-3) the <span class="term">[covariance](https://en.wikipedia.org/wiki/Covariance_and_correlation)</span> $\text{cov}(X,Y)=E[XY]-E[X]E[Y]$, and that for any linear regression model, regardless of the distribution of $X$,

$$ \text{cov}(X,Y) = \beta_1\sigma_X^2. $$

This gave us a way to *estimate* the slope from samples $(x_1,y_1),\dots,(x_n,y_n)$, without needing $X$ to be binary:

$$ \hat\beta_1 = \frac{\sum_{i=1}^n (x_i-\bar X)(y_i-\bar Y)}{\sum_{i=1}^n (x_i-\bar X)^2}, \qquad \hat\beta_0 = \overline Y - \hat\beta_1\overline X. $$

In Python, `np.cov(x,y)[0,1] / np.cov(x,y)[0,0]` computes $\hat\beta_1$ directly.

### Least squares interpretation

Suppose we plot the $(X,Y)$ points. Regardless of where they came from (a Normal model or not), we can compute $\hat\beta_1$ and $\hat\beta_0$ as above. These are known as <span class="term">[least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)</span> estimators, because it turns out they're exactly the values that minimize the sum of squared differences between the data points and the line $\hat\beta_1x+\hat\beta_0$ &mdash; i.e. they minimize the <span class="term">[residual sum of squares](https://en.wikipedia.org/wiki/Residual_sum_of_squares)</span> (RSS),

$$ \text{RSS} = \sum_{i=1}^n r_i^2, \qquad r_i = Y_i - (\hat\beta_1X_i+\hat\beta_0). $$

There are many other ways we could draw a line through a set of points; minimizing RSS happens to be the right choice under the assumption that the data come from a linear regression model, as above. In the demo below the vertical segments are the residuals $r_i$. Try to fit the line by eye first, then compare against the true minimizer of RSS.

{% include_relative demos/rss.html %}

<div class="example" markdown="1">
#### Example (marketing data)

Consider data on advertising budgets and sales for a company; we'll explore whether the TV advertising budget is associated with higher sales.

<u>Question:</u> fit the data to a linear regression model with the TV budget as the predictor and sales as the response.

<ol type="a">
<li>What are the estimates of $\beta_0$ and $\beta_1$?</li>
<li>Plot the regression line together with a scatter plot of the data.</li>
<li>Using the fitted $\hat\beta_0,\hat\beta_1$, simulate $10$ "fake" datasets with the same $x$ values and number of points as the real data, and compare their plots to the real data &mdash; does the model look reasonable?</li>
</ol>

<u>Solution:</u> see the companion Colab notebook.
</div>

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Estimating the regression slope by hand

Estimate the regression slope and intercept of the following dataset by hand:

<table class="prob-table">
<tr><th>$X$</th><th>$Y$</th></tr>
<tr><td>1</td><td>2</td></tr>
<tr><td>2</td><td>3</td></tr>
<tr><td>3</td><td>5</td></tr>
<tr><td>4</td><td>6</td></tr>
<tr><td>5</td><td>8</td></tr>
</table>

Then compute the residuals and RSS for your fitted line.
</div>

<div class="exercise" markdown="1">
#### Outlier sensitivity

Now add a new data point $(X_6,Y_6)=(100,3)$ to the dataset above. Explain, without doing a full calculation, why the fitted slope should change a lot. Then recompute the slope by hand and compare it to the original.
</div>

</details>

### Sample distribution of the estimator

$\hat\beta_1$ and $\hat\beta_0$ are computed from random data, so they are themselves random variables. If we collected a new dataset (same $x$ values, new draws of $Y$), we'd get slightly different estimates. This is the same idea as the sample distribution of $\overline Y$ from [Section 3.2](#sec-3-2), now applied to the regression slope.

To study this, we condition on the observed predictor values $x_1,\dots,x_n$, treating them as fixed constants &mdash; only the $Y_i$ are random. Writing $S_{xx}=\sum_{i=1}^n(x_i-\bar x)^2$, we can rewrite $\hat\beta_1$ as a weighted sum of the $Y_i$:

$$ \hat\beta_1 = \sum_{i=1}^n (x_i-\bar x)(Y_i-\bar Y)\big/S_{xx} = \sum_{i=1}^n w_iY_i, \qquad w_i = \frac{x_i-\bar x}{S_{xx}}, $$

using $\sum_i w_i(Y_i-\bar Y) = \sum_i w_iY_i - \bar Y\sum_i w_i = \sum_i w_iY_i$, since $\sum_i w_i=\sum_i(x_i-\bar x)/S_{xx}=0$.

<div class="example" markdown="1">
#### Example (unbiasedness of the slope estimator)

<u>Question:</u> show that $\hat\beta_1$ is unbiased, i.e. $E[\hat\beta_1]=\beta_1$.

<u>Solution:</u> since $x_i$ is fixed, $E[Y_i]=\beta_0+\beta_1x_i$. By linearity,

$$ E[\hat\beta_1] = \sum_{i=1}^n w_iE[Y_i] = \sum_{i=1}^n w_i(\beta_0+\beta_1x_i) = \beta_0\underbrace{\sum_i w_i}_{=0} + \beta_1\underbrace{\sum_i w_ix_i}_{=1}, $$

where $\sum_i w_ix_i=1$ because $\sum_i(x_i-\bar x)x_i = \sum_i(x_i-\bar x)^2+\bar x\sum_i(x_i-\bar x) = S_{xx}$, so $\sum_i w_ix_i = S_{xx}/S_{xx}=1$. Therefore

$$ E[\hat\beta_1] = \beta_1. $$

Across hypothetical repeated datasets with the same $x$'s, $\hat\beta_1$ is correct *on average*. (The same argument shows $E[\hat\beta_0]=\beta_0$.) One can similarly show $\operatorname{var}(\hat\beta_1) = \sigma^2/S_{xx}$ (we won't derive this): the more spread out the $x$'s, or the more data we have, the tighter the sample distribution of $\hat\beta_1$ is around $\beta_1$.
</div>

The demo below makes this concrete: the predictor values $x_1,\dots,x_n$ are fixed once and for all, and each click draws a *new* set of $Y_i$'s from the model and refits $\hat\beta_0,\hat\beta_1$. Every thin line is one hypothetical dataset's fitted line; the thick line is the true regression line. The cloud of fitted lines *is* the sample distribution of $(\hat\beta_0,\hat\beta_1)$, made visible.

{% include_relative demos/beta-sampling.html %}

## 3.5 Autoregressive models and the Hurwicz bias {#sec-3-5}

A special type of regression model where the predictor is the same quantity as the response variable but observed at a previous times is called an <span class="term">[autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model)</span>.  We consider a simple AR model where 

$$Y_t|Y_{t-1} \sim {\rm Normal}(\beta_1 Y_{t-1} +\beta_0,\sigma^2)$$

We further assume that $Y_t\mid Y_{t-1}$ is independent of $Y_{t-2},Y_{t-3},...$. This is a natural starting point for modeling for many noisy processes, such as stocks or height along a lineage (mother,daugher,granddaugher). The shows how the natural least squares estimato is biased. 

<div class="example" markdown="1">
#### Example

<u>Question:</u> Simulate many AR(1) series with true $\beta_1=0.6$ (and $\beta_0=0$), and estimate $\beta_1$ by least squares (regressing $Y_t$ on $Y_{t-1}$, no intercept) in each. Does the average of these estimates match $0.6$? What happens as the series gets longer?

<u>Solution:</u>

```python
import numpy as np

rng = np.random.default_rng(42)
beta1 = 0.6

def fit_beta1_hat(T):
    eps = rng.normal(size=T)
    Y = np.empty(T)
    Y[0] = eps[0]
    for t in range(1, T):
        Y[t] = beta1 * Y[t - 1] + eps[t]
    y_lag, y = Y[:-1], Y[1:]
    return np.sum(y_lag * y) / np.sum(y_lag**2)

for T in [10, 50, 200]:
    beta1_hats = [fit_beta1_hat(T) for _ in range(20_000)]
    print(f"T={T}: average beta1_hat = {np.mean(beta1_hats):.3f}")
```

which prints

| $T$ | average $\hat\beta_1$ |
|---|---|
| 10 | 0.506 |
| 50 | 0.577 |
| 200 | 0.595 |

At every $T$ the average estimate falls short of the true $\beta_1=0.6$, and the gap shrinks as $T$ grows: $\hat\beta_1$ is biased downward in any finite sample, but consistent.
</div>

The bias bound in the example above can be understood as follows. For the no-intercept AR(1) model, it can be shown (Hurwicz, 1950) that for large $T$,

$$ E[\hat\beta_1] - \beta_1 \approx -\frac{2\beta_1}{T}. $$

This matches the simulation: with $\beta_1=0.6$, the formula predicts a bias of about $-0.024$ at $T=50$ and $-0.006$ at $T=200$, both close to what we observed above (the match is worse at $T=10$, since the formula is only a leading-order approximation for large $T$). The bias vanishes as $T\to\infty$, consistent with $\hat\beta_1$ being a consistent estimator, but at any finite $T$ it pulls the estimate toward $0$ &mdash; least squares systematically underestimates how persistent the process really is. This finite-sample downward bias of the AR(1) least-squares estimator is known as the <span class="term">[Hurwicz bias](https://en.wikipedia.org/wiki/Autoregressive_model#Hurwicz_bias)</span>.

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### The stationary distribution of an AR(1) model

Consider the AR(1) model $Y_t\mid Y_{t-1}\sim\text{Normal}(\beta_1Y_{t-1}+\beta_0,\sigma^2)$ with $\lvert\beta_1\rvert<1$. As $t\to\infty$, the distribution of $Y_t$ settles down to a fixed <span class="term">[stationary distribution](https://en.wikipedia.org/wiki/Stationary_process)</span> that no longer depends on $t$ &mdash; that is, $Y_t$ and $Y_{t-1}$ have the same mean $\mu$ and the same variance $v$.

<ol type="a">
  <li>Using $E[Y_t]=E[Y_{t-1}]=\mu$ and taking the expectation of both sides of the model equation, write an equation for $\mu$ and solve for it in terms of $\beta_0,\beta_1$.</li>
  <li>Using $\operatorname{var}(Y_t)=\operatorname{var}(Y_{t-1})=v$, and that $Y_{t-1}$ and the noise term are independent, write an equation for $v$ and solve for it in terms of $\beta_1,\sigma^2$.</li>
  <li>Why do we need $\lvert\beta_1\rvert<1$ for a stationary distribution to exist? What happens to $v$ as $\beta_1\to1$?</li>
  <li>Simulate one long AR(1) series with $\beta_0=1,\beta_1=0.7,\sigma^2=4$, discard the first $500$ steps as "burn-in," and compare a histogram of the remaining values to the stationary Normal distribution you derived in (a) and (b).</li>
</ol>
</div>

</details>


## 3.6 Hypothesis testing {#sec-3-6}

In statistics we often infer parameters not because we care about their exact values, but because we want to use them to make a decision &mdash; e.g. in a clinical trial, whether a candidate drug is worth pursuing. This is often framed as <span class="term">[hypothesis testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)</span>: we assign a probability to a hypothesis (or its converse). In abstract terms, the basic procedure is:

<ol>
<li markdown="span">Come up with a <span class="term">[null hypothesis](https://en.wikipedia.org/wiki/Null_hypothesis)</span> &mdash; e.g. that the mean of some variable is zero &mdash; that we'd like to potentially rule out.</li>
<li markdown="span">Compute a <span class="term">[test statistic](https://en.wikipedia.org/wiki/Test_statistic)</span> $\hat T$: like any estimator, some quantity computed from the data.</li>
<li markdown="span">Ask a probabilistic question: what's the chance of observing a value of $\hat T$ at least as large as what we measured, <em>if the null hypothesis were true</em>? The result is the <span class="term">[$p$-value](https://en.wikipedia.org/wiki/P-value)</span>.</li>
</ol>

<div class="example" markdown="1">
#### Example (hypothesis testing for a binary-predictor regression model)

Consider a clinical trial again: an effect $Y$ (e.g. blood pressure) is measured in a control group ($X=0$) and a treatment group ($X=1$), with $N/2$ people in each group. Model

$$ Y\mid X \sim \text{Normal}\big(\mu_C(1-X)+\mu_TX,\ \sigma^2\big), $$

**assuming $\sigma^2$ is known** (this simplifies things a lot). This is a linear regression model, since

$$ \mu_C(1-X)+\mu_TX = \mu_C+(\mu_T-\mu_C)X = \beta_0+\beta_1X, \qquad \beta_0=\mu_C,\ \ \beta_1=\mu_T-\mu_C. $$

We could estimate $\beta_0,\beta_1$ as usual, or equivalently estimate $\mu_C,\mu_T$ directly within each group. Since $\sigma$ is known, the sample distributions are

$$ \hat\mu_C \sim \text{Normal}\Big(\mu_C,\ \frac{\sigma^2}{N/2}\Big), \qquad \hat\mu_T \sim \text{Normal}\Big(\mu_T,\ \frac{\sigma^2}{N/2}\Big), $$

so

$$ \hat\beta_1 \sim \text{Normal}\Big(\beta_1,\ \frac{4\sigma^2}{N}\Big). $$

Our null hypothesis is $\beta_1=0$ (no drug effect). As test statistic, we measure how far $\hat\beta_1$ is from zero in standard errors:

$$ \hat T = \frac{\hat\beta_1}{\text{se}(\hat\beta_1)}. $$

Since $\sigma$ is known, $\text{se}(\hat\beta_1)$ is known too, so from the sample distribution's perspective this is just division by a constant.

Let $\hat\beta_1^*$ denote the measured effect <em>under the null hypothesis</em> &mdash; a replicate generated assuming $\beta_1=0$. Then $\hat\beta_1^*$ has the sample distribution shifted to be centered at zero:

$$ \hat\beta_1^* \sim \text{Normal}\Big(0,\ \frac{4\sigma^2}{N}\Big). $$

We can now answer the question posed in step 3: if the null hypothesis were true, how likely would we be to see a $\lvert\hat T\rvert$ at least as large as what we observed? This is the $p$-value,

$$ p_v = P\big(\lvert\hat T^*\rvert > \lvert\hat T\rvert \mid \hat T\big), $$

where $\hat T^*$ is the test statistic computed from $\hat\beta_1^*$ (probability taken over its distribution), while $\hat T$ is fixed by our actual data.
</div>

The demo below shows both sides of this at once. On the left is the null distribution of $\hat\beta_1^*$ with the two-sided $p$-value shaded &mdash; the chance of seeing something at least as extreme as $\hat\beta_1$. On the right is the $95\%$ confidence interval for $\hat\beta_1$, drawn against $0$.

{% include_relative demos/pvalueCI.html %}

The example above is unusually simple ($\sigma$ known, binary predictor); in general computing $p$-values is more involved, but the principle and interpretation carry over. **Interpreting the $p$-value:** if $p_v$ is very small, it's highly unlikely we'd have seen our data if the null hypothesis were true, so we can *reject* the null hypothesis. Some threshold is usually chosen for this &mdash; conventionally, a result is <span class="term">[statistically significant](https://en.wikipedia.org/wiki/Statistical_significance)</span> if $p_v<0.05$. **If $p_v$ is not small, that does not mean the null hypothesis is true** &mdash; only that we failed to reject it. Visually, $\hat\beta_1$ is statistically significant exactly when $0$ is *not* in its confidence interval.

### $p$-values and confidence intervals

The $p$-value is about the "tail" of the sample distribution &mdash; its far ends. There's a natural connection to confidence intervals, which also measure the width of the sample distribution. In the simple Normal case above, testing $H_0:\beta_1=0$ at the $5\%$ level is equivalent to checking whether $0$ is outside the $95\%$ confidence interval for $\beta_1$. More generally, the two-sided $p$-value is the smallest significance level at which the null value would be excluded by the corresponding two-sided confidence interval.

</details>

## Problems {#problems}

<div class="exercise" markdown="1">
#### Problem 3.1 &mdash; Estimating a rate from waiting times

This problem introduces one new distribution, then uses it to show that an estimator built out of an unbiased estimator need not itself be unbiased.

<u>The model.</u> Suppose we time how long we wait for something to happen &mdash; a customer to arrive, an atom to decay, a bus to show up. A standard model for such a waiting time is the <span class="term">[exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)</span>, written $T \sim \text{Exponential}(\lambda)$. It is a continuous distribution on $[0,\infty)$ in the sense of [Section 1.4](../unit1/#sec-1-4), with density

$$ f(t) = \lambda e^{-\lambda t}, \qquad t \ge 0. $$

The single parameter $\lambda>0$ is a <em>rate</em>: large $\lambda$ means events arrive quickly, so waits are short. You do not need to derive anything about it; the two facts you need are

$$ E[T] = \frac1\lambda, \qquad \operatorname{var}(T) = \frac{1}{\lambda^2}, $$

and that `rng.exponential(scale=1/lam, size=n)` draws samples from it in `numpy` (note that `numpy` is parameterized by the mean $1/\lambda$, not by $\lambda$).

<u>The estimator.</u> Since $E[T]=1/\lambda$, we can estimate $E[T]$ by the sample average of measurements $T_1,\dots,T_n$,

$$ E[T] \approx \overline T = \frac1n\sum_{i=1}^n T_i, $$

and $\overline T$ is unbiased for $1/\lambda$ by linearity of expectation. Inverting suggests a natural estimator of the rate itself,

$$ \hat\lambda = \frac{1}{\overline T}. $$

<ol type="a">
<li>Before simulating anything: $\overline T$ is an unbiased estimator of $1/\lambda$. Does it follow that $1/\overline T$ is an unbiased estimator of $\lambda$? Say what you expect and why, in one or two sentences.</li>
<li>Show, using simulations, that $\hat\lambda$ is in fact a <em>biased</em> estimator of $\lambda$, although the bias decreases with $n$. To do this:
<ul>
<li>Make a list of $100$ values of $\lambda$ (any reasonable range works &mdash; e.g. between $0.2$ and $2$).</li>
<li>For each value of $\lambda$: simulate $10{,}000$ replicates of an experiment where each replicate has $n=5$ values of $T$; for each replicate compute $\hat\lambda$; then estimate $E[\hat\lambda]$ by averaging over replicates, and save it.</li>
<li>Plot $\lambda$ vs. $\lvert E[\hat\lambda]-\lambda\rvert$.</li>
</ul>
</li>
<li>Repeat part (b) with $n=50$ instead of $n=5$, on the same axes. Which direction does the bias go in, and roughly how does its size change?</li>
<li>(<strong>optional, ungraded</strong>) Consider the case $n=2$. Prove that
$$ E[\hat\lambda] = E\Big[\frac{1}{\overline T}\Big] \ge \lambda. $$
This is a special case of Jensen's inequality, and it is the general reason your answer to (a) is what it is: for a convex function $g$, $E[g(X)] \ge g(E[X])$.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 3.2 &mdash; Earnings data

Consider the earnings data, loaded with

```python
import pandas as pd

url = (
    "https://raw.githubusercontent.com/avehtari/"
    "ROS-Examples/master/Earnings/data/earnings.csv"
)
df = pd.read_csv(url)
male = df["male"].to_numpy()
earn = df["earn"].to_numpy()
earnk = df["earnk"].to_numpy()
height = df["height"].to_numpy()
```

pandas is used only to load the CSV; work with the numpy arrays `male`, `earn`, `earnk`, and `height` defined above (`statsmodels` accepts numpy arrays directly).

You'll study the association between earnings and gender, and how it depends on height. Later we'll see a better way to answer this question using regression with multiple predictors, but this more elementary approach highlights some key aspects of regression analysis.

<ol type="a">
<li>What do you expect the association between gender and earnings to be? Where do your expectations come from (news, intuition, other courses you've taken)?</li>
<li>Using <code>statsmodels</code>, perform a linear regression with gender (the array <code>male</code>) as the predictor and earnings as the response. You can use either <code>earnk</code> or <code>earn</code>, just keep track of units. Is there a statistically significant effect? Is the direction and size of the effect what you expected?</li>
<li>Using <code>statsmodels</code>, perform a linear regression with height as the predictor and earnings as the response. Answer the same questions as in part (b).</li>
<li>You should have found an association between both (gender, earnings) and (height, earnings). A natural question: is the height/earnings association simply a byproduct of men being taller on average? To check, separate the data into males and females and fit the height &rarr; earnings regression separately within each group.</li>
<li>Based on the previous part, what do you conclude? Is the association between height and earnings solely due to the association between gender and height, or does it look partially due to height itself?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 3.3 &mdash; Statistical significance (optional challenge)

Take the two-group setup of [Section 3.6](#sec-3-6) and write $\Delta\hat\mu = \hat\mu_T-\hat\mu_C = \hat\beta_1$ for the estimated treatment effect in one experiment. Show (using math or simulations) that it's possible to run two such experiments where $\Delta\hat\mu$ is statistically significant for one and not the other, yet the *difference* between the two experiments' $\Delta\hat\mu$ values is not itself statistically significant. Here "statistically significant" means $p<0.05$.
</div>

<div class="exercise" markdown="1">
#### Problem 3.4 &mdash; The random walk

The **random walk** is a foundational model across the sciences: it describes the "motion" of a variable that moves randomly over time with no memory of its past (Einstein used it to model microscopic particle motion, and it's a common rudimentary model of stock prices).

Let $X_0=0$ and define $X_k$ recursively by

$$ X_{k+1} = X_k + \Delta(2U_k-1), \qquad U_k \sim \text{Bernoulli}(1/2)\ \text{iid}, $$

where $\Delta$ is a constant. Think of $X_k$ as the position of someone randomly stepping left or right by $\Delta$ at each time step; the sequence $X_0,X_1,X_2,\dots$ is the walker's path.

<ol type="a">
<li>Write a Python function <code>simulaterw(Delta,K)</code> that simulates a random walk for $K$ steps, returning the whole path as a numpy array. Make some plots of $X_k$ vs. $k$.</li>
<li>What are $E[X_k\mid X_{k-1}=2]$ and $E[X_k]$?</li>
<li>Using the CLT, derive an approximation for the <strong>mean squared displacement</strong> $\text{MSD}(X_k)=E[X_k^2]$ (notice this is just another name for variance, in this context). Verify your approximation by plotting $\text{MSD}(X_k)$ as a function of $k$.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 3.5 &mdash; Normal approximation to an estimator

Consider $Y\mid X \sim \text{Bernoulli}(q_0+(q_1-q_0)X)$ with $X\in\lbrace 0,1 \rbrace$ &mdash; a model for a study with a control group ($X=0$) and treatment group ($X=1$); e.g. the treatment might be a heart medication and $Y=1$ a heart attack. Write $\Delta=q_1-q_0 = E[Y\mid X=1]-E[Y\mid X=0]$ for the difference in outcome probability between groups.

Suppose $N$ people are randomly assigned, $X\sim\text{Bernoulli}(1/2)$. We estimate $\Delta$ via the difference of sample averages:

$$ \widehat\Delta = \frac{N(Y=1,X=1)}{N(X=1)} - \frac{N(Y=1,X=0)}{N(X=0)}, $$

where, as in Unit 1, $N(\cdot)$ counts the rows satisfying a condition.

<ol type="a">
<li>Write a function <code>generate_data(q0, delta, n_samples)</code> producing a pair of numpy arrays $X,Y$, and a function <code>estimate_delta(X, Y)</code> producing $\widehat\Delta$ (you may adapt the code from the difference-of-means example in Unit 2).</li>
<li>Estimate the number of samples needed for a $95\%$ chance that the estimate is within $0.1$ of the true value.</li>
<li>Pick values for $q_0$ and $\Delta$, and test your result.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 3.6 &mdash; Laplace's rule of succession

Let $X\sim\text{Bernoulli}(q)$ (i.e. $P(X{=}1)=q$), with samples $X_1,\dots,X_N$. We've seen that $\hat q=Y/N$ (with $Y=\sum_iX_i$) is a consistent, unbiased estimator of $q$. An alternative, <span class="term">[Laplace's rule of succession](https://en.wikipedia.org/wiki/Rule_of_succession)</span>, is $\hat q_L=(Y+1)/(N+2)$.

The motivation: think of $X$ as a biased coin, and suppose we know it's possible to get either heads or tails. Using $\hat q$, a sequence of all heads (or all tails) gives $\hat q=1$ (or $0$) &mdash; ignoring the fact we know both outcomes are possible. To correct this, $\hat q_L$ pretends we've seen two additional observations, one heads and one tails (hence $Y+1$ over $N+2$). This is a simple example of incorporating *prior* information &mdash; here, that a coin has two sides, however unlikely one might seem &mdash; into an estimator.

<ol type="a">
<li>Derive $\text{MSE}_{\hat q_L} = E[(\hat q_L-q)^2]$ and decompose it into variance and squared bias, in terms of $q$ and $N$.</li>
<li>Is $\hat q_L$ unbiased? Is it consistent?</li>
<li>Now compute $\text{MSE}_{\hat q}$ (its bias is zero, so this should be straightforward from the standard error). Surprisingly, $\text{MSE}_{\hat q} > \text{MSE}_{\hat q_L}$ for some values of $N,q$ &mdash; for which values? This is surprising since $\hat q$ seems like it should be the best guess of $q$!</li>
</ol>
</div>
