---
layout: page
title: Course overview
---

**Instructor:** Ethan Levien  

**Prerequisites:** Some exposure to probability/statistics (e.g., Math 10, 20) and comfort with coding (e.g., CS 1, internships).  Also, see [Unit 0]({{ '/unit0/' | relative_url }}).

**Not registered?** Fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSfbbAGQl4w7lDcdOoEI2b3adkOBjpEox5hQIUx_cT6bW7gwAw/viewform).  

**Course objectives:**  

The course introduces core concepts in regression modeling through a combination of analytical derivations and numerical experimentation. Students develop a working understanding of how regression methods behave in practice, with an emphasis on bridging the gap between traditional statistics approaches and machine learning. 

Specific topics include: Conditional probability and expectation, basic statistical inference, single- and multiple-predictor linear regression models, autoregressive models, interactions, feature maps, prediction vs. inference, bias–variance trade-off, Bayesian linear regression, regularization and the kernel trick.


See [weekly schedule]({{ '/schedule/' | relative_url }}) for details.  
  

**My availability:**  

- Office (Kemeny 205)
- Office hours (in person): Tuesday 8:00–10:00am **and** 11:00am–12:30pm  
- If you would like to have a textbook for reference, I would recommend the following: 

**Textbooks:** My notes are mostly self contained, although I will reference material from a few textbooks: 
- James, Gareth, Witten, Daniela, Hastie, Trevor, Tibshirani, Robert, et al. (2013). *An introduction to statistical learning (python version)* (ISLP). Springer.  
  - This book straddles the boundary between machine learning and statistics and I will reference it heavily in Units 5-7. I found the discussion of regularization very approachable. 
- Gelman, Andrew, Hill, Jennifer, Vehtari, Aki. (2020). *Regression and other stories (ROS)*. Cambridge University Press. 
  - This is a fantastic, non-technical, introduction to regression modeling through coding and examples. Many aspects of the course are inspired by this book. The code is in R, but you can find versions in Python. It has a nice introduction to causal inference and I really like how they talk about interpreting regression coefficients. I also like that they take a Bayesian view, but are not too strict about it and focus on the practical goals of regression modeling, rather than the underlying philosophy. 
- Evans, Michael J., Rosenthal, Jeffrey S. (2004). *Probability and statistics: The science of uncertainty*. Macmillan.
  -  This is a more mathematically technical book on statistical inference and probability theory. If you are interested in going deeper into the theory, I recommend taking a look. I will reference it heavily in Units 1-4, but only for those who are interested in going deeper. 

You should be able to find PDFs of all these books online. 

**Software:**  

All coding will be done using Python in [Colab Notebooks](https://colab.research.google.com/). Within Python, we will use several packages throughout the course, including:  

- [numpy](https://numpy.org/) for arrays, linear algebra, and generating random numbers  
- [pandas](https://pandas.pydata.org/) for working with tabular data sets  
- [statsmodels](https://www.statsmodels.org/stable/index.html) for classical statistics  

**LLM Policy:** 

You are free to use LLMs at will on all take-home assignments, subject to the following: 
- All LLM output must be reviewed and edited. If I feel I am reading completely LLM-generated answers with long, padded sentences and LLM tropes, I reserve the right to deduct points, even if they are correct. 
- When you use an LLM for an assignment, you must include an *LLM disclosure statement* with the following information:
  - What models were used and how (e.g., Claude Code vs. a chat app).
  - A 1-2 paragraph reflection summarizing how the model was used: List the main contributions (e.g., pieces of code, making a specific plot) and how you engaged with those contributions to ensure the assignment was still beneficial.

Finally, unless otherwise indicated, it should be assumed that an assignment can be completed without an LLM.

You must also review the [full LLM policy]({{ '/ai-policy/ai_policy.pdf' | relative_url }}) (PDF) for the rationale behind this policy and practical recommendations.

**You are not permitted to use LLMs or any electronics during in-class exams or during lecture.**



## Assignments

Your grade will be based on the following. You should see Canvas for the detailed grading scheme and see the linked pages for assignment details.


**Exams (75%):** There will be three [exams]({{ '/exams/' | relative_url }}). Each midterm takes half a class period; the rest of that class is used for new material.  
- <u>Midterm 1</u> (20%): In class on September 30th, covering Units 1–2.  
- <u>Midterm 2</u> (20%): In class on October 28th, covering Units 3–4.  
- <u>Final</u> (35%): During finals week, November 25th at 3pm (room TBD), cumulative over Units 1–6. Your lower midterm score is replaced by the final if the final is higher. 


**Weekly assignments (15%):**  In addition to the practice problems, there will be a few more open ended assignments you will turn in each week.  

**Attendence and Participation (10%):** You are expected to be present in class ever day. While I will not check every student's attendence each class. I will randomly select a unspecified number of students and check if they are present each lecture. If you miss class or are late, I reserve the right to adjust your final grade. 

**Exercises (not graded):** Each section of the notes ends with a <u>Drill</u> dropdown, and each unit ends with a set of <u>Problems</u>. You should do all of them and ask questions if you have any. The drills are written at exam level &mdash; short calculations, interpreting a formula, reading a few lines of code &mdash; so they are the best preparation for the midterms and the final. 


## How to be successful in this course

- Come to class and ask questions.
- Do every exercise in the notes and every question on the practice exams.
- Use the data analysis competition as an opportunity to put the course material into practice on a concrete, hands-on task.
- Spend time reviewing the material and working through problems without LLMs/AI. These tools can create the illusion of productivity because of the rapid feedback.
- Connect with other students in the class and work with them. The class is typically very diverse: some students have taken other probability courses, like 20 and 40, while others have experience doing data analysis in a research or industry setting. It's great if you can connect with people whose skills complement yours.

## Accessibility Needs

Students with disabilities who may need disability-related academic adjustments and services for this course are encouraged to see me privately as early in the term as possible. Students requiring disability-related academic adjustments and services must consult the Student Accessibility Services office (Carson Hall, Suite 125, 646-9900). Once SAS has authorized services, students must show the originally signed SAS Services and Consent Form and/or a letter on SAS letterhead to me.  

As a first step, if you have questions about whether you qualify to receive academic adjustments and services, please contact the SAS office. All inquiries and discussions will remain confidential.  

## LLM Disclosure

Claude Code (various models including Sonnet 5 and  Opus 5) was used to edit and generate some material for Math 50, as well as to build this website. The original drafts of the unit notes were written without LLMs between 2021 and 2023. I prompted claude to convert my original latex files to markdown.  When doing this, I asked Claude to replace my hand-drawn figures in the original notes with interactive demos. I also prompted it to fix grammar and typos, and to create new drills and some new interactive demos, which I described in my prompt. In a few cases Claude added new text when converting the notes (such as instances of broken sentences). I either removed or heavily edited this text. Some of the drills claude wrote I was happy with and left almost entirely unchanged. When creating new drills, I usually give Claude a list of ideas for problems, but don't provide numerical values. I had Claude draft the sections on the Hurwicz bias and on seeding random number generators, which I ended up heavily editing. Claude also generated original drafts of slides based on selected sections of the notes, but these were almost entirely rewritten, except for some examples.