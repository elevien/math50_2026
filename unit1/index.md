---
layout: notes
title: "Unit 1: Probability Basics"
unit_title: "Unit 1"
unit_subtitle: "Probability models, conditioning, independence, simulation, and densities"
toc:
  - {href: "#sec-1-1", label: "1.1 Basic definitions"}
  - {href: "#sec-1-2", label: "1.2 Independence & conditioning"}
  - {href: "#sec-1-3", label: "1.3 Sampling and simulation"}
  - {href: "#sec-1-4", label: "1.4 Continuous distributions"}
  - {href: "#sec-1-5", label: "1.5 Binomial dist.", optional: true}
  - {href: "#problems", label: "Problems"}
---

<div class="unit-overview" markdown="1">

This unit introduces **probability models** as a framework for reasoning about uncertainty in data. The primary goal is to learn the mathematical language and notation needed to talk about statistical models (probability theory). We will start with probability models where the outcomes are a finite set (like the roll of a die). We'll then extend the framework to **continuous** random variables and probability densities which we will use for most of the course. 

#### Concepts

Probability models, random variables, sample spaces, outcomes and events, probability distributions, Bernoulli random variables, joint, marginal, and conditional probabilities, Bayes' formula, independence, iid samples, simulation as an approximation to probability, continuous random variables and probability density functions, the uniform distribution, and (optional) the binomial distribution.

#### Things to practice

- Move fluently between representations of the same model: words, tables, formulas, diagrams, simulations, and empirical counts from data.
- Translate between mathematical definitions and code: recognize where sample spaces, probabilities, conditioning, independence assumptions, and repeated sampling appear in a simulation (on exams you will not be asked to write code, but you may be asked to explain what a few lines of code are doing).
- Compute and interpret joint, marginal, and conditional probabilities, then use them to decide whether variables are independent.
- Explain the difference between exact probability statements, estimates from finite samples, and subjective uncertainty about a single unknown outcome.
- Reason about continuous random variables using intervals, densities, and conditioning, without treating density values themselves as probabilities.

</div>

<p class="pdf-link"><a href="unit1.pdf">Unit 1 notes pdf</a> <a href="https://colab.research.google.com/github/elevien/math50_2026/blob/main/unit1/unit1.ipynb">Unit 1 notebook</a></p>

The first step in our journey into regression modeling is to establish a language and system of notation for communicating uncertainty. Before doing so, let's say a few words about modeling in general. Broadly speaking, <span class="term">[models](https://en.wikipedia.org/wiki/Scientific_modelling)</span> are simplified representations of the world &mdash; astrology is a model of human behavior, and Newton's laws are a model of how objects move in physical space. Neither is perfectly correct, but Newton's laws provide a remarkably good approximation. In science (and in life) we mostly rely on <span class="term">[mathematical models](https://en.wikipedia.org/wiki/Mathematical_model)</span>.

The subject of this course is <span class="term">[regression models](https://en.wikipedia.org/wiki/Regression_analysis)</span>. We will define these precisely later, but roughly speaking, a regression model describes how the distribution of a variable $y$ (the *response*) is related to another variable $x$ (the *predictor*). Examples include predicting height from age, predicting disease risk from a genetic mutation, or a large language model predicting the next word in a sentence.

We will mostly study *linear* regression models, which in mathematical notation take the form

$$ y = \sum_{i} \beta_i x_i + \text{"noise."} $$

This equation says that $y$ &mdash; the quantity we want to predict &mdash; is a sum of observed variables plus some randomness. For instance, suppose we want to predict how long someone will live. Even knowing their entire medical history, and their parents', there is always some irreducible uncertainty.

## 1.1 Basic definitions {#sec-1-1}

### Sample spaces, random variables, outcomes and events

Loosely speaking, <span class="term">[random variable](https://en.wikipedia.org/wiki/Random_variable)</span> is any variable we cannot predict prior to observing it, no matter how much information we have &mdash; like the outcome of a coin flip. We use capital letters, often $X, Y, Z$, to denote random variables.

A random variable (or collection of them) has a <span class="term">[sample space](https://en.wikipedia.org/wiki/Sample_space)</span>, denoted $S$: the set of all values it can take. We write $S_X$ when we want to indicate the variable explicitly, and drop the subscript when it's clear from context. For a coin flip, $S = \lbrace \text{heads}, \text{tails}\rbrace$ &mdash; usually we represent these numerically, e.g. $1$ and $0$. For a die roll, $S = \lbrace 1,2,3,4,5,6\rbrace$. For the height of a tree, $S = \mathbb{R}_{\ge 0}$, although very tall trees become vanishingly unlikely, so we could just as well restrict to $[0, 1000\,\text{ft}]$. More on continuous sample spaces later in [Section 1.4](#sec-1-4). For now we'll stick to finite sample spaces.

The elements of $S$ are called outcomes and subsets of outcomes are called <span class="term">[events](https://en.wikipedia.org/wiki/Event_(probability_theory))</span>. For example, getting heads on a coin flip is an outcome (and an event). Getting greater than $2$ on a dice roll is an event. For an event $U$, we write $P(U)$ as its probability. 

Formally, a probability measure $P$ on a sample space $S$ satisfies the following axioms:

- **Nonnegativity:** for every event $U \subseteq S$, $P(U) \ge 0$.
- **Normalization:** $P(S) = 1$.
- **additivity:** for any countable collection (meaning we can list them as $U_1,U_2,\dots$) of pairwise disjoint sets $\lbrace U_i\rbrace_{i=1}^\infty \subseteq S$, $\displaystyle P\Big(\bigcup_{i=1}^\infty U_i\Big) = \sum_{i=1}^\infty P(U_i)$.


In particular, by additivity, 
$$ P(U) = \sum_{x \in U} P(\{x\}) $$. 

A direct consequence of these axioms is the <span class="term">[complement rule](https://en.wikipedia.org/wiki/Complementary_event)</span>. If $U^c = S \setminus U$ denotes the complement of an event $U$ (everything *not* in $U$), then $U$ and $U^c$ are disjoint and $U \cup U^c = S$, so additivity and normalization give $P(U) + P(U^c) = P(S) = 1$, i.e.

$$ P(U^c) = 1 - P(U). $$

This is often the easiest way to compute the probability of an "at least one" event: rather than adding up every way at least one thing happens, compute the probability that *none* of them happen and subtract from $1$.

Probabilities $P(\{x\})$ (that is, probabilities of outcomes) are called the  <span class="term">[probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)</span>. We'll write this in equivalent ways:

$$ P_X(x) = P(\{x\}) = P(X=x) = \text{chance that } X=x \text{ for } x \in S_X $$

or just $P(x)$ when it's clear we mean $X$. It's important to be fluid between different notations. 


<div class="example" markdown="1">
#### Example (Bernoulli distribution)

The <span class="term">[Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)</span> is probably the simplest random variable: it models a binary outcome, e.g. a YES/NO survey response or a diagnostic test result. If $Y$ follows a Bernoulli distribution, then

$$ P_Y(y) = \begin{cases} 1-q & y=0 \\ q & y=1 \end{cases} $$

Remember that $P_Y(y) = P(Y=y) = P(\lbrace Y=y\rbrace)$. These formulas make sense for any $0 \le q \le 1$; we call $q$ a <span class="term">[parameter](https://en.wikipedia.org/wiki/Statistical_parameter)</span> of the distribution. Instead of writing out the formula every time, we write

$$ Y \sim \text{Bernoulli}(q). $$

</div>

For any event $E$, we can define a random variable $1_E$ which is $1$ if event $E$ happens and $0$ otherwise &mdash; so the probability of an event can always be phrased as the probability of a Bernoulli random variable.

In general, for a random variable with a particular name and set of parameters, we write

$$ \text{Variable} \sim \text{Distribution}(\text{parameters}). $$

Not all random variables have specific names, but when they do, this notation lets us avoid writing out a probability distribution explicitly. Moreover, for more complicated random variables we can often express them in terms of random variables with known names.

<div class="example" markdown="1">
#### Example (joint distribution of two diagnoses)

Suppose we sample two people independently from the population and, for each, record whether they are diagnosed with condition $X$. For now, we will understand ``independent'' to mean that knowing one person's diagnosis does not change the probabilities for the other person's diagnosis. Model each person's diagnosis as an independent $\text{Bernoulli}(p)$ random variable with $(0,1) = (\text{no diagnosis}, \text{diagnosis})$, where $p$ is the (unknown, population-wide) chance of a diagnosis before $50$. Together the two people give a pair $(X_1, X_2)$ with sample space

$$ S = \lbrace (0,0), (0,1), (1,0), (1,1)\rbrace, $$

and some distribution $P_{X_1,X_2}(x_1,x_2)$, which we call the <span class="term">[joint distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution)</span> of $X_1$ and $X_2$. Because the two people are independent, the chance of any particular pair is just the chance of the first diagnosis times the chance of the second. 

$$ P(0,0) = (1-p)^2, \quad P(0,1) = (1-p)p, \quad P(1,0) = p(1-p), \quad P(1,1) = p^2. $$

</div>

The tree below builds these four probabilities branch by branch. Each path from the root multiplies the probabilities along its edges. Drag the slider to change $p$ and ensure you understand how to calculate the numbers of the leaves. 

{% include_relative demos/tree.html %}






<div class="example" markdown="1">
#### Example (colon cancer diagnosis and gender)

Let $C \in \lbrace 0,1 \rbrace$ indicate whether a randomly chosen person is diagnosed with colorectal cancer before age $50$, and let $G \in \lbrace M, F\rbrace$ be their gender. Combining the roughly even split of the U.S. population by gender with the cumulative before-age-$50$ diagnosis rates reported by the [American Cancer Society](https://www.cancer.org/content/dam/cancer-org/research/cancer-facts-and-statistics/annual-cancer-facts-and-figures/2024/mr7-probability-by-age-2024.pdf) &mdash; $0.42\%$ of men (about $1$ in $239$) and $0.38\%$ of women (about $1$ in $265$) &mdash; gives the joint distribution

<table class="prob-table">
<tr><th>$P(C,G)$</th><th>$G=M$</th><th>$G=F$</th></tr>
<tr><td>$C=1$ (diagnosed)</td><td>0.0021</td><td>0.0019</td></tr>
<tr><td>$C=0$ (not diagnosed)</td><td>0.4899</td><td>0.5061</td></tr>
</table>

A few natural questions about this table foreshadow ideas we'll make precise in the next section:

- *How likely is a randomly chosen person to be diagnosed, regardless of gender?* To address this we add across $M$ to obtain  $P(C=1) = 0.0021+0.0019 = 0.004$ or $0.4\%$ overall. Summing over one variable like this is called <span class="term">marginalizing</span> it.
- *Does knowing someone's gender change their risk?* 
- *Are $C$ and $G$ independent?* These conditional risks are close to each other and to the overall $0.4\%$, so before age $50$ gender barely affects cancer risk. 
</div>

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Drill 1 &mdash; Sample spaces and events

For each situation, identify tbe random variable(s) involeved, their sample space, and give examples of some events.

<ol type="a">
  <li>A randomly selected student answers YES or NO on a survey.</li>
  <li>A die is rolled until the first $6$ appears.</li>
  <li>A person from a study is selected and we record their height and gender. </li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 2 &mdash; Probabilities from a tree diagram

Two people are independently and randomly selected from a population in which the chance any given person supports a ballot measure is $0.6$.

<ol type="a">
  <li>Draw the probability tree for the pair of responses (support / not support), labeling every branch and every leaf with its probability.</li>
  <li>What is the probability that both people support the measure?</li>
  <li>What is the probability that exactly one of the two supports the measure?</li>
  <li>What is the probability that at least one of the two supports the measure?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 3 &mdash; Sample spaces of compound random variables

For each scenario, work out the sample space $S_Y$ of the described random variable $Y$.

<ol type="a">
  <li>A fair coin is flipped and a fair six-sided die is rolled. $Y$ is the die roll plus $1$ if the coin lands heads, or the die roll plus $4$ if the coin lands tails.</li>
  <li>A spinner is equally likely to land red, green, or blue. If it lands red, a coin is flipped and $Y$ is $0$ (tails) or $1$ (heads); if it lands green or blue, a fair four-sided die is rolled and $Y$ is the value shown.</li>
  <li>A fair six-sided die is rolled. $Y$ is the value shown if it's odd, or twice the value shown if it's even.</li>
  <li>For the scenario in (a), is every value in $S_Y$ reached by only one (coin, die) combination? If not, name a value that can arise in more than one way, and give both combinations that produce it.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 4 &mdash; Getting caught skipping class

There are $N=35$ students in the class. Each day, $5$ students are selected uniformly at random and checked for attendance. You skip $1/2$ of all classes, independently from day to day and independently of who gets checked. There are $M=27$ class meetings this term (see the [schedule]({{ '/schedule/' | relative_url }})).

<ol type="a">
  <li>What is the probability that you are one of the $5$ students checked on a given day?</li>
  <li>Let $X$ be the event that you are checked <i>and</i> absent on a given day &mdash; that is, you get caught skipping. Find $P(X)$.</li>
  <li>Treating the outcome on each of the $M$ days as independent, what is the probability that you are never caught all term?</li>
  <li>What is the probability that you get caught at least once (and thus lose participation points)?</li>
</ol>
</div>

</details>

## 1.2 Independence, conditioning and marginal distributions {#sec-1-2}

### Independence and marginal distributions

Two random variables $X, Y$ are <span class="term">[independent](https://en.wikipedia.org/wiki/Independence_(probability_theory))</span> if

$$ P(X=x, Y=y) = P(X=x) P(Y=y) $$

for all $x \in S_X$ and $y \in S_Y$. This will make more sense once we introduce conditional probability, below.

<div class="example" markdown="1">
#### Example (gene model)

Let's again consider two random variables taking values in $\lbrace 0,1\rbrace$. To be concrete: $Y_A$ and $Y_B$ represent whether an individual has a mutation on genes $A$ and $B$. Suppose we're given the joint distribution

$$ P(Y_A=y_A, Y_B=y_B) = \begin{cases} 1/2 & y_A=0,\, y_B=0 \\ 1/8 & y_A=0,\, y_B=1 \\ 1/8 & y_A=1,\, y_B=0 \\ 1/4 & y_A=1,\, y_B=1 \end{cases} $$

The sample space is the same four-outcome space as the cancer-diagnosis example above, but we can check that $Y_A$ and $Y_B$ are *not* independent:

$$ P(Y_A=0, Y_B=0) = \frac12 $$

whereas

$$ P(Y_A=0) = P(Y_A=0,Y_B=1) + P(Y_A=0,Y_B=0) = \frac18 + \frac12 = \frac58 $$

$$ P(Y_B=0) = P(Y_A=0,Y_B=0) + P(Y_A=1,Y_B=0) = \frac12 + \frac18 = \frac58 $$

and

$$ P(Y_A=0)P(Y_B=0) = 25/64 \approx 0.39 \ne 1/2. $$

</div>

The procedure above, summing over the other variable, is called <span class="term">[marginalization](https://en.wikipedia.org/wiki/Marginal_distribution)</span>. In general, for a joint distribution $P(x,y)$, the marginal distribution of $x$ is

$$ P(x) = P_X(x) = \sum_{y \in S_Y} P(x,y). $$

Notice that for independent variables,

$$ \sum_{y \in S_Y} P(x,y) = P(x) \sum_{y \in S_Y} P(y) = P(x). $$

The table below starts out as the gene-mutation joint distribution from the example above. Its margins are the sums $P(y_A)=\sum_{y_B}P(y_A,y_B)$; conditioning divides a row by its margin, which is the definition $P(y_A\mid y_B)=P(y_A,y_B)/P(y_B)$ in table form. Before clicking anything, try computing $P(Y_A=1)$, $P(Y_B=0)$ and $P(Y_A=1 \mid Y_B=0)$ yourself, then check your answers, You can regenerate the table for a fresh one to practice on.

{% include_relative demos/joint.html %}

<div class="example" markdown="1">
#### Example (calculating probabilities from a specified joint distribution)

Suppose $A,B,C \in \lbrace 0,1\rbrace$ have joint distribution $p(a,b,c) = P(A=a,B=b,C=c)$ given by:

<table class="prob-table">
<tr><th>$(a,b,c)$</th><th>$p(a,b,c)$</th></tr>
<tr><td>$(0,0,0)$</td><td>0.1</td></tr>
<tr><td>$(0,0,1)$</td><td>0.2</td></tr>
<tr><td>$(0,1,0)$</td><td>0.1</td></tr>
<tr><td>$(0,1,1)$</td><td>0.1</td></tr>
<tr><td>$(1,0,0)$</td><td>0.1</td></tr>
<tr><td>$(1,0,1)$</td><td>0.1</td></tr>
<tr><td>$(1,1,0)$</td><td>0.1</td></tr>
<tr><td>$(1,1,1)$</td><td>0.2</td></tr>
</table>

From this table we can compute many probabilities by adding the relevant rows:

$$ P(C=1) = \sum_{a}\sum_{b} p(a,b,1) = p(0,0,1)+p(0,1,1)+p(1,0,1)+p(1,1,1) = 0.2+0.1+0.1+0.2 = 0.6. $$

Similarly,

$$ P(A=1) = p(1,0,0)+p(1,0,1)+p(1,1,0)+p(1,1,1) = 0.5. $$

We can also compute probabilities involving more than one variable. For example,

$$ P(A=0,B=1) = p(0,1,0)+p(0,1,1) = 0.2, $$

and the event that $A$ and $B$ take the same value is

$$ \lbrace (0,0,0),(0,0,1),(1,1,0),(1,1,1) \rbrace, $$

so

$$ P(A=B) = 0.1+0.2+0.1+0.2 = 0.6. $$

The same table can be used to check independence. Since

$$ P(A=1,C=1) = p(1,0,1)+p(1,1,1) = 0.3, $$

while

$$ P(A=1)P(C=1) = 0.5 \times 0.6 = 0.3, $$

this particular cell is consistent with independence between $A$ and $C$.However, to prove independent we would need to check the product rule for every possible pair of values.

</div>

<div class="example" markdown="1">
#### Example (colon cancer, continued: marginalizing the other way)

Revisiting the colon cancer and gender table from Section 1.1, we can marginalize out $C$ instead of $G$ to recover the gender split:

$$ P(G=M) = P(C=0,G=M) + P(C=1,G=M) = 0.4899 + 0.0021 = 0.492, $$
$$ P(G=F) = P(C=0,G=F) + P(C=1,G=F) = 0.5061 + 0.0019 = 0.508. $$

We can also check independence formally, using the product-rule definition above, rather than just eyeballing the numbers as we did in Section 1.1:

$$ P(C=1,G=M) = 0.0021, \qquad P(C=1)P(G=M) = 0.004 \times 0.492 = 0.001968. $$

These are close but not exactly equal. What do we make to this? Is this is real dependence or a property of our finite data set? This will be addressed later. 
</div>



### Conditioning

What if we're interested in the chance someone has a mutation in gene $A$, given we know they don't have one in gene $B$? We write this as the <span class="term">[conditional probability](https://en.wikipedia.org/wiki/Conditional_probability)</span> $P(Y_A=1 \mid Y_B=0)$: the chance gene $A$ is mutated, restricted to people without a mutation in gene $B$. More generally, $P(X \mid Y=y)$ is the distribution of $X$ once we know $Y=y$, assuming $P(Y=y)>0$.

It's useful to write this in terms of joint and marginal probabilities. If $N(E)$ denotes the number of times event $E$ occurs in a dataset of $n$ total observations, then

$$ P(Y_A=1 \mid Y_B=0) = \frac{N(Y_A=1, Y_B=0)}{N(Y_B=0)} = \frac{N(Y_A=1,Y_B=0)/n}{N(Y_B=0)/n} = \frac{P(Y_A=1, Y_B=0)}{P(Y_B=0)}. $$

<div class="example" markdown="1">
#### Example (gene model, continued)

In the gene model above, the conditional probability of $Y_A=1$ given $Y_B=0$ is

$$ P(Y_A=1 \mid Y_B=0) = \frac{P(1,0)}{P(Y_B=0)} = \frac{1/8}{5/8} = \frac15. $$

Try it in the demo above: click "condition on $Y_B=0$" and compare.
</div>

We use the notation $Y \mid (X=x)$ for the random variable $Y$ conditioned on $X=x$; this has a distribution that depends on $x$. As shorthand we might write $Y \mid X$, treating the distribution as a function of $X$.

<div class="example" markdown="1">
#### Example (conditional Bernoulli model)

Suppose $Y \sim \text{Bernoulli}(1/2)$, and $X \mid Y \sim \text{Bernoulli}(Y/4 + 1/4)$. That is: first sample $Y$; then, given $Y$, sample $X$ from $\text{Bernoulli}(Y/4+1/4)$. As a table:

<table class="prob-table">
<tr><th>$y$</th><th>$P(Y=y)$</th></tr>
<tr><td>0</td><td>1/2</td></tr>
<tr><td>1</td><td>1/2</td></tr>
</table>
<table class="prob-table">
<tr><th>$x$</th><th>$y$</th><th>$P(X=x \mid Y=y)$</th></tr>
<tr><td>1</td><td>0</td><td>1/4</td></tr>
<tr><td>0</td><td>0</td><td>3/4</td></tr>
<tr><td>1</td><td>1</td><td>1/2</td></tr>
<tr><td>0</td><td>1</td><td>1/2</td></tr>
</table>

The joint probability is then

$$ P(X=x,Y=y) = P(X=x\mid Y=y)\,P(Y=y). $$

</div>

In general,

$$ P(x\mid y) = \frac{P(x,y)}{P(y)}, \qquad \text{when } P(y)>0. $$

Replacing $P(x,y)$ with $P(y\mid x)P(x)$ gives <span class="term">[Bayes' formula](https://en.wikipedia.org/wiki/Bayes%27_theorem)</span>:

$$ P(x\mid y) = \frac{P(y\mid x)P(x)}{P(y)}. $$

<div class="example" markdown="1">
#### Example (colon cancer, continued: conditioning the other way)

In Section 1.1 we computed $P(C=1\mid G=M) \approx 0.0043$: the chance of an early diagnosis, given someone is male. Bayes' formula lets us flip the conditioning around and ask the reverse question &mdash; given someone was diagnosed before $50$, what's the chance they're male?

$$ P(G=M \mid C=1) = \frac{P(C=1\mid G=M)\,P(G=M)}{P(C=1)} = \frac{0.0043 \times 0.492}{0.004} \approx 0.53, $$

which matches computing it directly from the table, $P(C=1,G=M)/P(C=1) = 0.0021/0.004 = 0.525$, up to rounding. Notice how different this is from $P(C=1\mid G=M)\approx0.0043$: conditioning is not symmetric, $P(A\mid B)$ and $P(B\mid A)$ can be worlds apart, even though they're built from the same joint distribution.
</div>

An equivalent definition of independence is $P(y\mid x) = P(y)$ and $P(x\mid y) = P(x)$ whenever the conditional probabilities are defined. In summary:

$$ X \text{ and } Y \text{ independent} \iff P(x,y)=P(x)P(y) $$

$$ \iff P(y\mid x)=P(y) \iff P(x\mid y)=P(x). $$

Bayes' formula also holds for events &mdash; for example $P(\lbrace Y>y\rbrace\mid X=x) = P(\lbrace Y>y\rbrace,X=x)/P(X=x)$, since $\lbrace Y>y\rbrace$ can always be represented as a Bernoulli random variable.

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Drill 5 &mdash; Conditional probability from a joint table

The joint probability table for $A$ and $B$ is:

<table class="prob-table">
<tr><th></th><th>$B=0$</th><th>$B=1$</th></tr>
<tr><th>$A=0$</th><td>0.1</td><td>0.3</td></tr>
<tr><th>$A=1$</th><td>0.2</td><td>0.4</td></tr>
</table>

<ol type="a">
  <li>What is $P(A=1)$?</li>
  <li>What is $P(B=1)$?</li>
  <li>What is $P(A=1 \mid B=1)$?</li>
  <li>What is $P(B=0 \mid A=0)$?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 6 &mdash; Marginalization from a joint distribution

Let $X$ and $Y$ have joint probabilities $P(X=0,Y=0)=0.2$, $P(X=0,Y=1)=0.2$, $P(X=1,Y=0)=0.1$, $P(X=1,Y=1)=0.5$.

<ol type="a">
  <li>Compute the marginal distribution $P_X(x)$ for $x=0,1$.</li>
  <li>Compute the marginal distribution $P_Y(y)$ for $y=0,1$.</li>
  <li>Compute $P(Y=1 \mid X=0)$.</li>
  <li>Compute $P(X=1 \mid Y=0)$.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 7 &mdash; Conditional probability from a piecewise model

Let $Z$ have $P_Z(0)=0.2$, $P_Z(1)=0.5$, $P_Z(2)=0.3$. Let $W \mid Z \sim \text{Bernoulli}(Z/4+1/4)$.

<ol type="a">
  <li>What is $P(W=1 \mid Z=2)$?</li>
  <li>What is $P(W=1)$?</li>
  <li>What is $P(Z=1 \mid W=1)$?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 8 &mdash; Marginalization and conditioning with three variables

Using the same $P(A=a,B=b,C=c)$ table as the worked example above:

<ol type="a">
  <li>Compute $P(A=1)$.</li>
  <li>Compute $P(B=1 \mid A=0)$.</li>
  <li>Compute $P(C=1 \mid A=1, B=1)$.</li>
  <li>Compute $P(B=0)$.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 9 &mdash; Independence from a model

Consider the model

$$ Y_B \sim \text{Bernoulli}(2/3), \qquad Y_A \mid (Y_B=0) \sim \text{Bernoulli}(1/3), \qquad Y_A \mid (Y_B=1) \sim \text{Bernoulli}(1/2). $$

<ol type="a">
  <li>Write down the joint probability distribution of $Y_A$ and $Y_B$.</li>
  <li>What are the marginal distributions of $Y_A$ and $Y_B$?</li>
  <li>Are $Y_A$ and $Y_B$ independent? Explain using the product rule or conditional distributions.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 10 &mdash; Reverse-engineering a joint distribution

Suppose $X,Y \in \lbrace 0,1,2\rbrace$ are independent, with marginal distributions

$$ P_X(0)=0.2,\ P_X(1)=0.5,\ P_X(2)=0.3, \qquad P_Y(0)=0.4,\ P_Y(1)=0.4,\ P_Y(2)=0.2. $$

<ol type="a">
  <li>Use independence to construct the full joint distribution $P(X=x,Y=y)$ for all nine pairs $(x,y)$.</li>
  <li>Now perturb your table: add $\varepsilon=0.05$ to $P(X=0,Y=0)$ and to $P(X=1,Y=1)$, and subtract $\varepsilon=0.05$ from $P(X=0,Y=1)$ and from $P(X=1,Y=0)$, leaving every other entry as in part (a). Check that the row sums still give $P_X$ and the column sums still give $P_Y$.</li>
  <li>Are $X$ and $Y$ independent under this new table? Justify your answer using the product rule.</li>
  <li>In one sentence, explain what parts (b) and (c) show about whether a joint distribution is determined by its marginals.</li>
</ol>
</div>

</details>

## 1.3 Sampling and simulation {#sec-1-3}

### IID samples and simulations

A measurement of a random variable is a <span class="term">[sample](https://en.wikipedia.org/wiki/Sample_(statistics))</span>, and <span class="term">[statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)</span> is the process of estimating the parameters $\theta$ of a distribution from a sample. In statistics we seek to answer questions like:

- Suppose we don't have information about every student in the college. A survey of five students from this class finds $4$ YESes and $1$ NO. What is our best prediction of the fraction of students in the whole college who would say YES? What assumption are we making when we answer this?
- How many experiments do we need to run to know whether a drug is effective?

We'll discuss statistical inference in detail in Unit 3. For now, we usually assume (though it's not strictly true) that we're given independent samples of the same random variable. We call this <span class="term">[iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)</span> data (independent and identically distributed): each sample has the same distribution, and the value of any particular sample has no influence on the others. This is the situation when flipping a coin repeatedly, for example.

### From probabilities to samples

We now note the basic relationship between a probability and a sample: if we have $n$ samples of a random variable and outcome $x \in S_X$ occurred $N(x)$ times, then

$$ P(X=x) \approx \frac{N(x)}{n}. $$

Going forward we'll use $N$ to denote the number of times an outcome occurs. As with probability, we'll use a few equivalent notations for the number of samples where $X = x \in S$:

$$ N(\lbrace X=x\rbrace) = N(X=x) = N_X(x) = N(x). $$

### Python basics: numpy arrays

Before working through the code examples below, it's worth getting comfortable with a few basic `numpy` array operations we'll rely on throughout the course.

```python
import numpy as np

x = np.array([3, 1, 4, 1, 5])
y = np.array([1, 9, 4, 1, 5])

x[0]                     # 3          (first element)
x[1:3]                   # [1, 4]     (a slice)
x[x > 2]                 # [3, 4, 5]  (boolean indexing)
x[y <= 1]                # [3, 1]     (boolean indexing using another array)
x[(x == 1) & (y == 1)]   # [1]        (combine conditions with &)
```

`x[condition]`, where `condition` is a boolean array like `x > 2`, selects the entries where `condition` is `True`; this only works because `x` is a numpy array, not a plain Python list. Combine conditions with `&` (and) or `|` (or), and always use parentheses: since `&` binds tighter than `==`, writing `x == 1 & y == 1` (without the parentheses around each comparison) raises an error instead of doing what you'd expect.

A few more operations that come up constantly:

```python
len(x)        # 5           (number of elements)
np.sum(x)     # 14          (sum of elements)
np.mean(x)    # 2.8         (average, equal to np.sum(x) / len(x))

x + y         # [4, 10, 8, 2, 10]   (elementwise sum)
x * y         # [3, 9, 16, 1, 25]  (elementwise product)
```

Elementwise operations like `x + y` and `x * y` require `x` and `y` to have the same length.

<div class="example" markdown="1">
#### Example (estimating a coin's bias)

Suppose we flip a (possibly biased) coin $1000$ times and record the outcomes. Each flip is Bernoulli, $X \sim \text{Bernoulli}(p)$, where $p$ is the probability of heads. We don't know $p$ in advance, but we can estimate it from the observed data &mdash; for instance, if a survey asks each respondent YES/NO, the proportion of YES answers is our best estimate of the true support rate.

Below is Python code that simulates coin flips and uses the sample frequency to estimate $p$:

```python
import numpy as np

# True probability of YES (e.g. support for a policy)
p_true = 0.6

# Collect 1000 samples (YES=1, NO=0)
n = 1000
samples = np.random.choice([0, 1], p=[1-p_true, p_true], size=n)

# Estimate probability from sample frequency
p_hat = np.mean(samples)

print("Estimated probability:", p_hat)
```

On running this code, the output will be close to $0.6$, but not exactly, since the data are random. The difference between the estimate $\hat p$ and the true parameter $p$ is the central problem of statistical inference.
</div>

The simulator below runs this experiment. 

{% include_relative demos/bernoulli.html %}

<div class="example" markdown="1">
#### Example (Python: sampling and conditional probability from the gene model)

```python
import numpy as np

# Define probabilities for each (Y_A, Y_B) pair
probs = [1/2, 1/8, 1/8, 1/4]  # (0,0), (0,1), (1,0), (1,1)
pairs = [(0,0), (0,1), (1,0), (1,1)]

# Generate samples
N = 10000
samples = np.random.choice(len(pairs), size=N, p=probs)
Y_A = np.array([pairs[i][0] for i in samples])
Y_B = np.array([pairs[i][1] for i in samples])

# Compute conditional probability P(Y_A = 1 | Y_B = 0)
mask = (Y_B == 0)
cond_prob = np.mean(Y_A[mask] == 1)
print("P(Y_A = 1 | Y_B = 0) =", cond_prob)
```

This simulates the joint distribution, selects samples where $Y_B=0$, and computes the fraction of those where $Y_A=1$, thus it estimates $P(Y_A=1 \mid Y_B=0)$.
</div>

<div class="example" markdown="1">
#### Example (Python: DataFrame estimate of a conditional probability)

<u>Question:</u> using the same distribution as the $A,B,C$ example above, make a DataFrame with columns `A`, `B`, `C` whose rows are iid samples, then estimate $P(A=0 \mid B=1, C=1)$, both by hand and in code.

<u>Solution:</u> by definition,

$$ P(A=0 \mid B=1,C=1) = \frac{P(A=0,B=1,C=1)}{P(B=1,C=1)} = \frac{0.1}{0.1+0.2} = \frac{0.1}{0.3} = \frac13 \approx 0.3333. $$

```python
import numpy as np
import pandas as pd

# Support and probabilities (match the table above)
outcomes = np.array([
    (0,0,0), (0,0,1), (0,1,0), (0,1,1),
    (1,0,0), (1,0,1), (1,1,0), (1,1,1)
], dtype=int)
probs = np.array([0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2], dtype=float)

# Draw N iid samples
N = 200_000
idx = np.random.choice(len(outcomes), size=N, p=probs)
samples = outcomes[idx]

# Build DataFrame with columns A, B, C
df = pd.DataFrame(samples, columns=["A", "B", "C"])

# Estimate P(A=0 | B=1, C=1)
mask = (df["B"]==1) & (df["C"]==1)
est = (df.loc[mask, "A"]==0).mean()
print("Estimated P(A=0 | B=1, C=1):", est)
```

The printed estimate should be close to the exact value $1/3$ above.
</div>

### Python cheat sheet: sampling and probability from data

The examples above all reuse the same handful of moves: draw samples, index into arrays or DataFrames with a condition, and average a boolean condition to get a probability. Here's a reference for these moves. For the exams, you should be able to read a few lines like these and say what probability they're computing.

Throughout, `X` and `Y` are parallel 1D arrays of $N$ paired samples (e.g. `X[i], Y[i]` is the $i$th draw of $(X,Y)$), and `x`, `y` are specific values in their sample spaces.

| Task | Python |
|---|---|
| Draw $N$ samples, from the sample space `vals`, probs `p` | `X = np.random.choice(vals, p=p, size=N)` |
| Samples satisfying a condition | `X[X == x]` |
| Count of samples satisfying a condition | `np.sum(X == x)` |
| Marginal probability $P(X=x)$ | `np.mean(X == x)` |
| Joint probability $P(X=x, Y=y)$ | `np.mean((X == x) & (Y == y))` |
| Conditional probability $P(X=x \mid Y=y)$ | `np.mean(X[Y == y] == x)` |


The same computations look like this on a DataFrame `df` with columns `"X"`, `"Y"`, whose rows are the paired samples:

| Task | Python |
|---|---|
| Select one column | `df["X"]` |
| Select the rows satisfying a condition | `df.loc[df["Y"] == y]` |
| Select one column, restricted to rows satisfying a condition | `df.loc[df["Y"] == y, "X"]` |
| Combine two conditions (AND) | `df.loc[(df["Y"] == y) & (df["Z"] == z)]` |
| Marginal probability $P(X=x)$ | `(df["X"] == x).mean()` |
| Joint probability $P(X=x, Y=y)$ | `((df["X"] == x) & (df["Y"] == y)).mean()` |
| Conditional probability $P(X=x \mid Y=y)$ | `(df.loc[df["Y"] == y, "X"] == x).mean()` |

A probability is always the *fraction of rows satisfying a condition*, i.e. `.mean()` of a boolean array or column; conditioning just means restricting to a subset of rows first with `[...]` or `.loc[...]` before you take that average.

The DataFrame table is shown so you can *read* pandas code when it appears in examples or datasets. You don't need to know this for exams. 

### Seeding random number generators

The `np.random` calls above aren't truly random.  They run a deterministic algorithm that produces a sequence of numbers designed to *look* random. We call this a <span class="term">[pseudorandom number generator](https://en.wikipedia.org/wiki/Pseudorandom_number_generator)</span> (PRNG). A PRNG starts from a number called the <span class="term">[seed](https://en.wikipedia.org/wiki/Random_seed)</span>, and the entire sequence of "random" draws it produces is completely determined by that seed. The same seed reproduces the exact same numbers each time the code is run.

This is useful because simulations should be <span class="term">[reproducible](https://en.wikipedia.org/wiki/Reproducibility)</span>. If you're debugging a simulation, comparing two models on "the same" simulated data, or sharing code to check your work, you want the randomness to be fixed.

In `numpy`, the recommended way to do this is to create an explicit generator object with `np.random.default_rng(seed)` and draw from that object, as in

```python
rng = np.random.default_rng(123)
samples = rng.choice([0, 1], p=[0.4, 0.6], size=1000)
```

rather than the older, unseeded global functions like `np.random.choice(...)` used above. Any fixed integer works as a seed; changing it gives a different, but equally valid, run of the simulation, which is a good way to check that a result isn't an accident of one particular seed.

#### Note on probabilities as frequency vs. belief

Consider the coin from the example above, but suppose it's already been flipped and is now covered by a cup. Heads or tails is already decided, just hidden from us. If someone says "there's a 90% probability it landed heads," what does that number actually refer to?

One reading treats probability as a *frequency*: if we imagine repeating the flip (or repeating "situations like this one") many times, heads would come up in 90% of them. This is the interpretation behind the sampling picture above, where $N(x)/n \to P(X=x)$ as $n$ grows.

But this particular coin isn't going to be reflipped. Rather, it already landed, and the outcome is fixed, even though we don't know it. A second reading treats the 90% instead as a *degree of belief*: a number summarizing how confident we are that it's heads, given whatever evidence we have (maybe we saw it wobble, or we know this coin is biased). Under this view, probability doesn't require imagining repeated trials at all and therefore can quantify uncertainty about a single, already-determined fact.

Both readings obey the same rules of probability, so for the mathematics we've developed so far it won't matter which one you have in mind.

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Drill 11 &mdash; Reading simulation code

Consider the code below.

```python
p = 0.3
samples = np.random.choice([0, 1], p=[1-p, p], size=1000)
estimate = np.mean(samples)
```

<ol type="a">
  <li>What probability model is being sampled?</li>
  <li>What is the sample space of each draw?</li>
  <li>What probability is `estimate` approximating?</li>
  <li>If you ran the code twice, should `estimate` be exactly the same both times? Explain.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 12 &mdash; Translating code into a joint model

Consider the code below.

```python
y = np.random.choice([0, 1], p=[0.4, 0.6])
if y == 0:
    x = np.random.choice([0, 1], p=[0.8, 0.2])
else:
    x = np.random.choice([0, 1], p=[0.3, 0.7])
```

<ol type="a">
  <li>Write the model in notation using $Y$ and $X\mid Y$.</li>
  <li>Compute $P(X=1,Y=0)$ and $P(X=1,Y=1)$.</li>
  <li>Compute $P(X=1)$.</li>
  <li>Are $X$ and $Y$ independent?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 13 &mdash; Describing a simulation plan

Without writing exact code, describe how you would use simulation to check the probability you computed in Drill 9.

<ol type="a">
  <li>What random variables would each simulated row contain?</li>
  <li>How would you estimate a marginal probability from the simulated rows?</li>
  <li>How would you estimate a conditional probability from the simulated rows?</li>
  <li>What would you compare to decide whether the simulation agrees with your exact calculation?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 14 &mdash; Reading NumPy array code

Consider the array `x = np.array([4, 7, 7, 1, 3])`.

<ol type="a">
  <li>What is `x[2]`?</li>
  <li>What is `x[x >= 5]`?</li>
  <li>What is `np.mean(x == 7)`?</li>
  <li>What is `len(x[x > 3])`?</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 15 &mdash; Marginal, joint, and conditional probability from arrays

Suppose `x` and `y` are numpy arrays of $N$ paired iid samples of $(X,Y)$, where $X,Y \in \{0,1\}$.

<ol type="a">
  <li>Write a numpy expression that estimates $P(X=1)$.</li>
  <li>Write a numpy expression that estimates $P(X=1, Y=0)$.</li>
  <li>Write a numpy expression that estimates $P(Y=0 \mid X=1)$.</li>
  <li>Suppose `np.mean(x == 1)` is $0.45$ and `np.mean((x == 1) & (y == 0))` is $0.18$. What is the resulting estimate of $P(Y=0 \mid X=1)$?</li>
</ol>
</div>

</details>

## 1.4 Continuous distributions {#sec-1-4}

So far we've worked entirely with *discrete* sample spaces &mdash; ones we can list out, like $\lbrace 0,1\rbrace$ or $\lbrace 1,2,3,\dots\rbrace$. Many quantities we care about are naturally continuous instead: height, time, a stock price. We also run into quantities that are nearly continuous when they are built from many discrete ingredients. For iid Bernoulli $X_1,\dots,X_n$, the sample average $\bar X$ (the sum $X_1+\cdots+X_n$, divided by $n$) has sample space

$$ S_{\bar X} = \lbrace 0,\tfrac1n,\tfrac2n,\dots,1 \rbrace. $$

As $n$ grows, these possible values get closer and closer together inside $[0,1]$. This is not the same as a genuinely continuous random variable, but it motivates why we need language for distributions whose possible values form an interval. Fortunately, much of the discrete theory carries over once we replace sums with integrals &mdash; you won't be asked to evaluate integrals in this course, but it's worth seeing where the definitions come from.

### The uniform distribution

A simple starting point: $Y \sim \text{Uniform}(a,b)$, meaning $Y$ is equally likely to land anywhere in $[a,b]$. If $L=b-a$, then for $y_1 < y_2$ in $[a,b]$,

$$ P(y_1 \le Y \le y_2) = \frac{y_2-y_1}{L}. $$

The probability of landing in an interval is proportional to its length, which ensures $P(a\le Y\le b)=1$. As $y_2\to y_1$, this probability goes to zero &mdash; so $P(Y=y)=0$ for any specific $y$: there are uncountably many possible outcomes in any interval, so no single point can carry positive probability. 



### Densities

This motivates the general notion of a <span class="term">[probability density function](https://en.wikipedia.org/wiki/Probability_density_function)</span> (pdf). A continuous random variable $Y$ is characterized by a nonnegative function $f(y)$, its density, such that for any $a<b$,

$$ P(a < Y < b) = \int_a^b f(y)\,dy. $$

Geometrically, the probability is the area under $f(y)$ between $a$ and $b$. For a small interval of width $\Delta y$, $P(y \le Y \le y+\Delta y) \approx f(y)\,\Delta y$, so $f(y)$ plays the role of "probability per unit length." For the uniform case,

$$ f(y) = \begin{cases} 1/L & y\in[a,b] \\ 0 & \text{otherwise.} \end{cases} $$

Every pdf must satisfy:

- **Nonnegativity:** $f(y)\ge0$ for all $y$.
- **Normalization:** $\int_{-\infty}^\infty f(y)\,dy = 1$.

These are the continuous analogues of the discrete axioms. Note that $f(y)$ need *not* be less than $1$, because $f(y)$ is not itself a probability &mdash; only the integral $\int_a^b f(y)\,dy$ is. For example, if $Y$ is uniform on $[0,1/1000]$, then $f(y)=1000$ on that interval: the density is concentrated in a tiny region, but the large values it takes there cancel out so that integrals stay $\le 1$.

You won't have to calculate integrals in this class, but it's worth understanding where this comes from. Below is a interactive plot illustrating the idea of a density

{% include_relative demos/uniform.html %}



<div class="example" markdown="1">
#### Example (waiting for a bus)

Let $T$ be how long (in minutes) you wait for a bus, modeled as $T \sim \text{Uniform}(0,5)$.

<u>Question:</u> restricting to the cases where you've already waited more than $3$ minutes, what's the density of $T\mid(T>3)$? Check with simulation.

<u>Solution:</u> starting from the definition,

$$ P(t_1<T<t_2\mid T>3) = \frac{P(t_1<T<t_2,\,T>3)}{P(T>3)}. $$

Assuming $3<t_1$ and $t_2<5$, the numerator is $P(t_1<T<t_2) = (t_2-t_1)/5$ &mdash; since $T\in[t_1,t_2]$ already implies $T>3$, the chance both hold is just the chance of the more restrictive event. The denominator is $P(T>3) = (5-3)/5 = 2/5$, so

$$ P(t_1<T<t_2\mid T>3) = \frac{(t_2-t_1)/5}{2/5} = \frac{t_2-t_1}{2}, $$

meaning the density is $f(t\mid T>3) = 1/2$, i.e. $T\mid(T>3) \sim \text{Uniform}(3,5)$: conditioning a uniform on a sub-interval just gives a uniform on that sub-interval. In particular, $E[T\mid T>3] = (3+5)/2 = 4$ minutes.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)

N = 200_000
T = rng.uniform(0, 5, size=N)
T_cond = T[T > 3]

print(f"Proportion kept (should be ~2/5 ~ 0.400): {len(T_cond)/N:.3f}")
plt.figure(figsize=(6,4))
plt.hist(T_cond, bins=40, density=True, alpha=0.7, label="Simulated density")
plt.axhline(1/2, color="red", linestyle="--", label="Theoretical density f(t|T>3)=1/2")
plt.xlabel("Waiting time given T>3 (t)")
plt.ylabel("Density")
plt.legend()
plt.show()
```
</div>

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Drill 16 &mdash; Uniform probabilities

Let $Y \sim \text{Uniform}(0,10)$.

<ol type="a">
  <li>Compute $P(2<Y<5)$.</li>
  <li>Compute $P(Y>7)$.</li>
  <li>Compute $P(Y=4)$.</li>
  <li>Explain why the answer to part (c) is not a contradiction: values near $4$ can still be likely even though the exact value $4$ has probability zero.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 17 &mdash; Density is not probability

Suppose $Y$ is uniform on $[0,0.2]$.

<ol type="a">
  <li>What is the density $f(y)$ on this interval?</li>
  <li>Why is it acceptable for this density to be greater than $1$?</li>
  <li>Compute $P(0.05<Y<0.10)$.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 18 &mdash; Conditioning on an interval

Let $Y \sim \text{Uniform}(0,1)$. Compute each probability.

<ol type="a">
  <li>$P(Y<0.3)$</li>
  <li>$P(Y<0.3 \mid Y<0.5)$</li>
  <li>$P(Y>0.7 \mid Y>0.4)$</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 19 &mdash; Mixing a Bernoulli and a Uniform

Let $X \sim \text{Bernoulli}(0.3)$ indicate whether it rains today ($X=1$) or not ($X=0$). Your train's delay $Y$, in minutes, depends on the weather:

$$ Y \mid (X=0) \sim \text{Uniform}(0,5), \qquad Y \mid (X=1) \sim \text{Uniform}(0,20). $$

<ol type="a">
  <li>Compute $P(Y<3 \mid X=0)$ and $P(Y<3 \mid X=1)$.</li>
  <li>Use part (a) to marginalize out $X$ and compute $P(Y<3)$.</li>
  <li>Use Bayes' formula and parts (a)&ndash;(b) to compute $P(X=1 \mid Y<3)$.</li>
  <li>Without computing anything, explain in one sentence why your answer to (c) is smaller than $P(X=1)=0.3$.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 20 &mdash; Simulating a mixed model

Continuing the model from Drill 19, write code to simulate $N=200{,}000$ pairs $(X,Y)$, then use the samples to check your answers.

<ol type="a">
  <li>Simulate $X$ from its Bernoulli distribution. Then simulate $Y$, using boolean indexing (as in the <a href="#sec-1-3">Section 1.3 cheat sheet</a>) so that the samples with $X=0$ get $\text{Uniform}(0,5)$ draws and the samples with $X=1$ get $\text{Uniform}(0,20)$ draws.</li>
  <li>Estimate $P(Y<3)$ with <code>np.mean(...)</code> and compare it to Drill 19(b).</li>
  <li>Restrict to the samples with $Y<3$, then estimate $P(X=1\mid Y<3)$ from that subset. Compare it to Drill 19(c).</li>
</ol>
</div>

</details>

<details class="optional-section" id="sec-1-5" open markdown="1">
<summary><h2>1.5 Binomial distribution</h2> <span class="optional-badge">(optional)</span></summary>

Suppose $Y_i \sim \text{Bernoulli}(q)$, $i=1,\dots,N$, are independent, with $Y_i=1$ with probability $q$. Let

$$ Y = \sum_{i=1}^N Y_i. $$

Then $Y$ follows a <span class="term">[binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution)</span>, written $Y \sim \text{Binomial}(N,q)$.

<div class="example" markdown="1">
#### Example (calculating probabilities)

Let $N=3$, and ask: what is $P(Y=2)$?

<u>Solution:</u> there are 3 sequences that give $Y=2$: $(1,0,1), (1,1,0), (0,1,1)$. The probability of any one of these, e.g.

$$ P(y_1=1,y_2=0,y_3=1) = P(y_1=1)P(y_2=0)P(y_3=1) = q(1-q)q = q^2(1-q), $$

is the same for all three. So

$$ P(Y=2) = 3\,q^2(1-q). $$

</div>

The binomial distribution has two parameters, $N$ and $q$: the number of flips, and the probability of success. The chance of any *particular* configuration of $k$ ones is $q^k(1-q)^{N-k}$, by independence. But we need to account for the number of configurations with $k$ ones, which is

$$ \binom{N}{k} = \frac{N!}{k!(N-k)!} = \frac{N \times (N-1) \times \cdots \times (N-k+1)}{k \times (k-1) \times \cdots \times 1}. $$

<details markdown="1">
<summary>Proof</summary>

Let $C_{N,k}$ denote the number of length-$N$ sequences with $k$ ones. Split on the first entry: if it's a $1$, there are $N-1$ remaining slots for $k-1$ ones, giving $C_{N-1,k-1}$ sequences; if it's a $0$, there are $N-1$ slots for all $k$ ones, giving $C_{N-1,k}$. So

$$ C_{N,k} = C_{N-1,k-1} + C_{N-1,k}, $$

with $C_{N,0}=C_{N,N}=1$ as base cases (only one way to configure all zeros or all ones). Solving this recursion gives the formula above.
</details>

This implies

$$ P(Y=k) = \binom{N}{k} q^k (1-q)^{N-k}, $$

which looks like a bell curve when $N$ is large and $q$ is not too close to $0$ or $1$. The Python code below verifies this with Monte Carlo simulation, comparing a histogram of simulated draws to the analytical PMF:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# parameters
N = 100        # number of trials
q = 0.3        # success probability
M = 100000     # number of Monte Carlo replicates

rng = np.random.default_rng(123)
samples = rng.binomial(n=N, p=q, size=M)

bins = np.arange(-0.5, N + 1.5, 1)
k = np.arange(0, N + 1)
pmf = binom.pmf(k, N, q)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(samples, bins=bins, density=True, alpha=0.5, label="Monte Carlo (hist)")
(markerline, stemlines, baseline) = ax.stem(k, pmf, label="Analytical distribution")
plt.setp(baseline, visible=False)

ax.set_xlabel("k")
ax.set_ylabel("Probability")
ax.set_title(f"Binomial(N={N}, q={q})")
ax.legend()
ax.set_xlim(-0.5, N + 0.5)
plt.tight_layout()
plt.show()
```

The demo below does the same comparison interactively: the bars are the exact PMF $\binom{N}{k}q^k(1-q)^{N-k}$ and the outlines are Monte Carlo draws of $Y=\sum_i Y_i$. Raise $N$, and keep $q$ away from $0$ and $1$, to watch the bell shape emerge.

{% include_relative demos/binomial.html %}

<details class="practice-section" markdown="1">
<summary><h3>Drill</h3></summary>

<div class="exercise" markdown="1">
#### Drill 21 &mdash; Binomial probabilities

Let $Y \sim \text{Binomial}(4,q)$.

<ol type="a">
  <li>Write out the sample space of $Y$.</li>
  <li>Compute $P(Y=0)$.</li>
  <li>Compute $P(Y=1)$.</li>
  <li>Compute $P(Y\ge 1)$ in the simplest way you can.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Drill 22 &mdash; Counting configurations

Let $Y \sim \text{Binomial}(5,q)$.

<ol type="a">
  <li>How many sequences of $5$ flips give $Y=2$? Write $\binom52$ and evaluate it.</li>
  <li>What is the probability of one particular such sequence, e.g. $(1,1,0,0,0)$?</li>
  <li>Combine (a) and (b) to write $P(Y=2)$.</li>
  <li>Explain in one sentence why $P(Y=2)$ is <em>not</em> just $q^2(1-q)^3$.</li>
</ol>
</div>

</details>

</details>

## Problems {#problems}

<div class="exercise" id="ex-washpost" markdown="1">
#### Problem 1.1 &mdash; Homicide Victim Age and Race

Below we load data on homicide victims in the US from the Washington Post. You don't need to worry about how the file is processed; pandas is used only to read the CSV &mdash; just work with the numpy arrays `age` and `race` defined on the last two lines.

```python
import numpy as np
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/washingtonpost/data-homicides/master/homicide-data.csv", encoding="ISO-8859-1")
age = pd.to_numeric(df["victim_age"], errors="coerce").to_numpy()
race = df["victim_race"].to_numpy()
```

In this problem, treat a randomly selected row (i.e. a matched pair `age[i], race[i]`) as one outcome. Let $A$ be the victim's age. Among rows with known race, let $R$ indicate whether the victim is listed as white or not white.

<ol type="a">
  <li>Identify the sample space you are actually using. What rows, if any, should be excluded before computing probabilities involving age or race?</li>
  <li>Write code that computes $P(A<z)$ for each $z=1,\dots,100$ and plots the result. Before looking at the plot, sketch what shape you expect and explain why.</li>
  <li>Modify the code so it computes and plots $P(A<z \mid R=\text{white})$ and $P(A<z \mid R=\text{not white})$ on the same axes.</li>
  <li>Translate the two plotted curves into words: for a fixed value of $z$, what probability is each curve showing?</li>
  <li>Based on the plots, do age and race look independent in this dataset? Explain using conditional probability, not just visual language.</li>
  <li>Approximate $P(R=\text{white} \mid 10<A<60)$ from the data. Explain which rows are in the numerator and denominator.</li>
</ol>
</div>

<div class="exercise" markdown="1">
#### Problem 1.2 &mdash; Modeling Infection Risk After a Night Out

Suppose we're modeling how likely we are to contract SARS-CoV-3, a hypothetical more dangerous strain, after a night out, under these assumptions:

- We interact with exactly $N$ people in sequence, with no repeated interactions.
- Each person either has the virus or doesn't.
- $10\%$ of the student population has it.
- Given someone has it and we interact with them, there's a $50\%$ chance we contract it.
- Treat the outcomes of different interactions as independent.

<ol type="a">
  <li>Define a Bernoulli random variable for a single interaction that records whether that interaction infects us. What is its success probability?</li>
  <li>Let $G_N$ be the event that we get infected at least once over $N$ interactions. Justify the formula $P(G_N)=1-(1-x)^N$ and determine $x$.</li>
  <li>Write a simulation function that returns $1$ if we get infected at least once over $N$ interactions and $0$ otherwise. Then explain, line by line, how the code represents the model assumptions.</li>
  <li>Generate a plot comparing the exact probability $1-(1-x)^N$ with a Monte Carlo estimate for $N=1,2,\dots,50$.</li>
  <li>If the simulation and formula disagree substantially, what are three things you would check in the code or model assumptions?</li>
  <li>Change one assumption in the model, such as the infection probability per contact or the population prevalence. State the new probability model before running any code.</li>
</ol>
</div>
