---
layout: post
title: My Interview Experience at Tredence for DS Role
date: 2026-08-10 21:00:00
toc:
  sidebar: left
mermaid:
  enabled: true
  zoomable: true
tikzjax: true
typograms: true
description: In this blog, I have shared my complete Tredence Data Scientist recruitment experience, starting from the pre-placement talk and online assessments to the technical and HR interviews. I have also covered the questions asked in DSA, SQL, ML, DL, NLP, GenAI, DBMS, system design, Python, and more. I’ve also shared my preparation strategy, resources I used, tips for the placement season, and my overall experience with the recruitment process.
tags: [Interview Experience, Tredence]
categories: Interview
pinned: true
---

# Introduction

Tredence came up for the Data Scientist role. The recruitment process was as follows: first, a pre-placement talk, then Online Assessment 1, Online Assessment 2, followed by 2 technical interview rounds, and finally 1 HR round.

Talking about the PPT, it was held offline at the Chemical Engineering Department seminar hall. It started at 9 AM in the morning and was for almost 2 hours. Then, around 11:30 AM, the first OA was held in LHC D offline, and it was for 90 minutes.

Talking about the OA, it had 38 questions, out of which 1 question was for DSA, 1 question was for SQL, and 2 subjective questions were based on Pandas, Python, and NumPy. The rest of the questions were MCQs based on Machine Learning, Deep Learning, Probability, and Statistics. In Statistics, the questions were mainly based on hypothesis testing and p-values.

There were also pseudocode-based questions. One pseudocode-based question was on LCS, and we had to tick the correct DP transition for LCS. Link to the LCS question: [link](https://leetcode.com/problems/longest-common-subsequence/)

Some of the questions were also based on GenAI and vector databases. There was 1 MCQ question based on Pancake Sort, where we had to predict the output for a given test case.

Link to the Pancake Sort question: [link](https://leetcode.com/problems/pancake-sorting/)

## DSA Question

This was really luck-based. Some people got really easy questions, while some got DP-based questions.

My question was:

Given three arrays `a`, `b`, and `c`, where `a[i]` is the satisfaction level, `b[i]` denotes calories, and `c[i]` denotes the cost of the ith dish, we have to find the minimum cost sum such that the sum of calories is less than `Y` and we have at least `X` satisfaction level. It is always guaranteed that there exists an answer. Return the minimum cost sum of the dishes which satisfy the above conditions.

I solved it using DP.

The DSA question had 10 test cases and carried 20 marks in total.

For some of the people, DSA was really easy, and the questions were like finding the alternate sum of a linked list and easy questions based on greedy and heap.

## SQL Question

My question was relatively easy compared to others. It was based on applying 2 joins on the tables and then 1 aggregate function with a window function, along with `GROUP BY` and `ORDER BY` for the final output.

Some of my friends were asked recursive CTE-based questions.

OA 1 extended up to 1:30 PM, and then we had a lunch break.


Around 2:00 PM, the shortlist for OA 2 came out, and around 23 students were shortlisted. The second OA was again held offline in LHC D and was for 45 minutes.

This OA was divided into 4 sections, mainly testing listening comprehension, grammar, and reading comprehension.


### Section 1

It consisted of around 18 questions, as far as I remember. We were given a 6–7 second audio, which we could listen to only once and could not pause. We then had to record the same sentence as a voice note.

### Section 2

This section was based on grammar. It included fill-in-the-blanks, appropriate articles, tense, verbs, error-based questions, changing voice, changing tense, etc. There were around 45 questions.

### Section 3

This was based on listening comprehension. We had to listen to an audio of around 1 minute and then answer 3 questions based on it. The audio could only be played once and could not be paused or replayed. There were around 16 questions in this section.

### Section 4

We were given around 3–4 topics and had to speak continuously for around 1 minute on the given topic. Topics included things like "Describe a moment you are proud of" and "Your favourite hobby."

Around 3 PM, I completed the test, and after around 30 minutes, the technical interviews started in CDC at around 3:45 PM.


# Technical Round 1

My first technical interview started around 4:30 PM.

The interviewer introduced himself and then asked me to introduce myself. He then went through my resume and asked me to explain my resume and work experience.

Based on my projects, he asked questions about:

* Multithreading vs multiprocessing
* Why OpenMP?
* True parallelism vs concurrency
* Vector databases and indexing
* Word embeddings, Word2Vec
* Tokenization, lemmatization, and other NLP concepts

Then he asked me a DSA question, which was mainly based on Pandas/DataFrames with a follow-up.

### Part 1: Dataset Generation

I was given a list of sentences and had to use **Pandas and NumPy** to generate a dataset of 100,000 rows containing:

```text
ID | Sentence | Cost | Length
```

* ID from 1 to 100000
* Randomly select a sentence
* Random cost between 0 and 100
* Length of the selected sentence

First, I had to explain my approach on paper and then code it in Python on an online IDE and print the first 5 rows.

### Part 2: Splitting the Dataset

I was then given an LLM context window `K` and had to split the DataFrame into the **minimum possible number of DataFrames**, such that the total sentence length in each split does not exceed `K`.

I also had to consider the minimum cost for each split.

Again, I first explained my approach on paper and then started coding. I completed the first part completely, but since time was running out, I could only code half of the second part before the interviewer asked me to move on.

### SQL

He then asked me two SQL questions.

The first was an MCQ where I was given 4 SQL statements and had to identify the correct one.

The second question gave me 2 tables with their primary and foreign key relationships and asked:

> Find each customer's total spending and rank the customers based on their total spending, from highest to lowest.

I solved it using:
JOIN  GROUP BY  SUM  RANK ORDER BY

### Core ML

He then asked some core ML questions such as:

* Vanishing gradient
* Overfitting
* Cost function for Linear Regression
* Cost function for Logistic Regression
* Precision, Recall, and F1 Score

I had to explain/write some of these on paper.

### Bonus System Design Question

He also asked a bonus system design question about augmenting a small Text-to-SQL dataset using the **Quilbot API**.

The API rephrased English sentences while keeping the SQL query unchanged. However, after around 30–40 requests, it sometimes started returning the original sentence, which could cause an infinite loop.

I came up with a queue-based solution to handle the API requests and avoid the infinite loop. The interviewer then corrected my approach and suggested using a multi-level queue to handle the requests more efficiently.

Overall, Round 1 was pretty chill. The interviewer was also very helpful and gave me hints and helped with code and syntax when needed.


# Technical Round 2

Just after my Round 1, after 2–3 minutes, I was called for Round 2.

It started with the same interviewer introducing himself, followed by my introduction. After that, he asked me why I moved from Civil to the AI/ML field.

Then, for around 20 minutes, he mostly grilled me on my resume and work experience. He asked me to explain the system architecture of the AI service. **Disclaimer: most of the questions in this round were based on my resume and work experience.**

Questions on my resume were like:

* Draw the architecture diagram of the AI service
* List out SQL models and relationships between tables
* How the backend and AI service communicate
* Types of databases: SQL and NoSQL
* Methods of indexing a database
* Authentication vs authorization
* Docker and Kubernetes basic commands
* Unix questions

I had to explain these on pen and paper.

Then, after the resume, he asked me questions related to core ML:

1. Explain Logistic Regression, cost function, hypothesis, sigmoid, etc.
2. Exploding gradient problem in RNNs, why we prefer LSTM, and what problems LSTM solves
3. Encoder-decoder architecture
4. What is multi-class multi-label classification?
5. Softmax vs sigmoid, cross-entropy loss
6. R² score and bias-variance tradeoff

Then there was a rapid-fire round on databases, ML, AI, Python lambda functions, and SQL. This was pretty chill.

At last, he asked me this LeetCode question:
[link](https://leetcode.com/problems/valid-palindrome-ii/description/)

I had to explain the approach, write the pseudocode, explain the intuition with a dry run on a test case, and discuss the edge cases.


# HR Round

After 10 minutes, I was called for the HR interview.

The HR started with my introduction, followed by questions about my hobbies and family background. The HR round was pretty chill.

Some of the questions I remember were:

1. Describe a moment where you were really proud of yourself, related to any academic or non-academic experience.
2. What is one skill/talent that you have which backfires sometimes?
3. Describe yourself in 3 words.
4. Describe an event where you had to resolve a conflict.

The HR interview was really chill, and the interviewer gave me around 20–30 seconds to think before answering.

The HR interview ended around 7:40 PM in the evening.

At this point, I was pretty tired. Starting with the PPT at 9 AM in the morning and continuing till the HR interview was really tiring.

The SPOCs then said that some of the interviews were scheduled for the next day, so the results would be announced once all the interviews were completed.

The next day, the results were announced, and 2 of us were selected for the DS FTE roles :)


# Your Way of Preparation

I started with AI/ML in my first year itself. For core ML and AI resources, I would suggest watching Andrew Ng's ML and DL courses on Coursera, or CS229 and CS230 on YouTube.

For DSA, I would suggest starting with Striver's A2Z sheet as soon as possible.

For SQL, solve the SQL Top 50 questions.

For DBMS, watch Gate Smashers or Love Babbar's DBMS notes.

Start as early as possible.


# Overall Experience

Overall, the experience was really good. Everything was organized and well planned. The interviewers were also really supportive, and communication was really easy.

The SPOCs also coordinated really well and were helpful regarding any queries.

Overall, it was a really good experience.


# Advice for the Particular Company

Honestly, for Tredence, I would say it depends on your panel of interviewers, but be mindful of core ML/DL concepts because they are a must.

Also, try getting hands-on experience with GenAI and RAG through projects or work, etc.

Be thorough with your resume and focus mostly on DSA, SQL, and core ML, along with GenAI concepts. They mainly asked NLP and system design questions.

Try building AI/ML projects, and for the OAs, I would suggest revising your ML/DL notes.


# General Tips for Placement Season

Getting placed early helps a lot with mental peace, but the placement season can be quite stressful, especially when you have to manage academics and placements at the same time.

My biggest advice would be to start preparing as early as possible. Have a clear idea of the roles you are targeting and prepare consistently every day instead of trying to cover everything at the last moment. Building a strong foundation early will help you a lot during OAs and interviews.

Try to give as many mock interviews** as possible and take feedback from your seniors. Also, get your resume reviewed and keep improving it based on the roles you are applying for.

Most importantly, don't get demotivated by rejections. You will get rejected from interviews, you might not clear some OAs, and sometimes you may feel that your preparation is not working. This is completely normal during placement season.

Trust the process, keep improving after every rejection, and stay consistent. Eventually, you will get placed.

