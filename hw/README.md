# Homework (Voice Anti-spoofing)

## Task

Implement and train a Countermeasure (CM) system on the Logical Access (LA) partition of the [ASVSpoof 2019 Dataset](https://datashare.ed.ac.uk/handle/10283/3336) ([Kaggle Link](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)). You may find the [ASVspoof 2019 evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf) useful.

> [!TIP]
> Downloading the dataset through Kaggle can be faster. If you are working directly in Kaggle, you can attach the dataset as Kaggle-Input for speed up and saving memory.

## Countermeasure systems

This time, we restrict our solution to the LCNN architecture.

> [!CAUTION]
> You cannot use implementations available on the internet in any way (including looking at them).

### LightCNN

Implement [LightCNN (LCCN)](https://arxiv.org/abs/1511.02683) following the Speech Technology Center [paper](https://arxiv.org/abs/1904.05576).

**Hints**:

1. Take training recipe and data preparation scheme from [this paper](https://arxiv.org/abs/2103.11326). Also, read the comparative study and think whether you should use A-Softmax or Cross-Entropy loss function.

2. Use STFT (FFT in the paper) as front-end. (Though others may work too.)

3. Put dropout layer before the final BatchNorm.

## Code and logs

You have two options for code:

1. Write your code in a clone of [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template): bonus $B=2.0$. In this case, you submit **a link to your github repository.**
2. Write your code in a `ipynb` notebook: bonus $B=0.0$. In this case, you submit **a link to a Google Colab / Kaggle notebook.**

> [!IMPORTANT]
> We do not accept option 2 if it looks messy. Use markdown to split your notebook in meaningful sections. Add some comments.

Besides, you need to prove that you indeed trained a model by showing logs via [WandB](https://wandb.ai/site) or [Comet ML](https://www.comet.com/site/). **Add all the required plots: losses on train/eval; metrics on eval, etc.** Attach your logs as a WandB/Comet Report.

> [!TIP]
> When training a model, it is a good idea to start from a one-batch test. Once you have a pipeline ready, choose a fixed batch from your dataset and train your model on this single batch for many steps. Evaluate on the same batch. If your pipeline is correct, your model should be able to overfit on this batch and achieve almost perfect loss and quality. Then, you can continue with the full train-split dataset training and evaluation on the evaluation set. Note that passing one-batch test does not mean that there are no errors in your pipeline, however, not passing it means that something is likely wrong.

> [!TIP]
> Since the task is just a more advanced variant of 2-class classification, we advise you to first go through the [Self-Study notebook](https://github.com/Blinorot/deep-learning-research/blob/summer_2026/self_study/SelfStudyPractice.ipynb) to get familiar with core concepts on a toy example.

### Notes on the PyTorch template and Implementation Bonus

To make yourself familiar with the PyTorch template, you can have a look at the tutorials in its `README`. Furthermore, [this lecture](https://youtu.be/cOA-_6a7wrQ) from the DLR course can be helpful.

You will not get any bonus if any of the following rules are not followed:

- Your code must be a variant of the provided PyTorch template. It cannot be just any `.py`-style project code.
- Your model must be trained using this template. You cannot train your model first and then write the project code. Similarly, the logs must come directly from the template.
- Add a clear README and installation guide to your repository. Use meaningful commits.

Advice:

- Try to read the template code line-by-line. Use doc-strings to understand what each function is doing. Once you do it, it will be much easier for you to work with it.
- Start by implementing a familiar task. For example, you can do simple image classification and then compare your code with an example branch from the template.
- You may use `HuggingFace` branch though it is even more advanced.

**Hint**: to run project style code in Colab\Kaggle, you can do the following:

```bash
!git clone https://YOUR_PRIVATE_TOKEN@github.com/USERNAME/REPO_ID

# change the root dir
%cd REPO_ID

# install packages
...

# run the code
!python3 train.py ...
```

Code configuration and `%%writefile` will help you to avoid unnecessary commits.

> [!IMPORTANT]
> Make sure to remove your private token from code before submitting. Moreover, never push your private token to your repository.

## Report

As a part of the mini-course official rule, you are required to write a short report at the end of the practice. Ask your Program advisors about the specific rules for your program. In general, you should include the following information:

- Introduction: What the mini-course was about and what task are you solving at the end (Deepfake Detection). Add some motivation behind the task.
- Description of the task / Methodology: Introduce metrics and main terminology. Describe the LCNN model. You can add a diagram of the architecture.
- Experimental Setup: Describe how have you trained the model. What learning rate, batch size, what input features, how many steps, etc. Justify your setup by referencing the corresponding literature.
- Results: Show results and some plots. **Do not take a screenshot from WandB or Comet ML**. This looks unprofessional. Instead, download logs as a CSV file and draw them clearly using `matplotlib`. Describe if the model is working and if the achieved results align with those reported in the literature. You can add a discussion on the failed attempts and what was changed to make the model work.
- Conclusion: Summarize what have you learnt during the mini-course, indicate the main challenges you had to tackle, and conclude whether your model training was successful. If your model training was not successful, you may reflect on why this happened.

> [!CAUTION]
> Failing to provide a clear report can lead to zeroing your grade. Each section is more than just one line. Make sure that your sentences are connected to each other and the whole text makes sense. Format your text. Use formal language. In general, write text in such a way that it is clear for other people and not only for the author alone.

## Grade

Your mark ($M$) will depend on your model performance ($P$) and the bonus $B$ (based on the code-style you chose):

$$M = \min(P + B, 10)$$

As mentioned in the [Report section](#report), your grade can be decreased if the quality of your report is too low, though the report quality does not impact your grade above a certain threshold. This means that we do not expect you to write a perfect report, but we expect it to be of acceptable quality.

> [!IMPORTANT]
> Wandb / Comet ML logs are not graded but their absence or if they look suspicious will result in zero grade (your EER plot should slowly go closer and closer to your final EER). Likewise, if we find out that your solution is taken from the web or is AI-based, your grade will be zeroed.

Performance metrics (must be calculated on the **evaluation** set of the LA partition of the ASVspoof 2019 Dataset):

- Equal Error Rate (EER).

[Code](./calculate_eer.py) for metrics is provided in this repo (see `compute_eer` function). You should provide metrics in your report and achieve the following performance:

- EER range (**measured in % on 0-100 scale**):
  - $10.9 < EER$: $P=0$.
  - $5.3 <= EER <= 10.9\\%$: $P=2$ (linearly scales from $2$ at EER=$10.9$, to $10$ at $5.3$)
  - $EER < 5.3$: $P=10$

**Note**: requested EERs are much higher than the ones described in the paper to save your time.

Time to achieve full $P$ grade in Kaggle for the teacher's solution (may vary a bit if the random seed is bad):

| Model | Time (h) |
| ----- | -------- |
| LCNN  | 4.5-5    |

## Submission

To submit your solution, use the Google Form provided in the course channel. Indicate:

- Meta-information, including your Name, official university email address, etc.
- Submission URL:
  - If your solution is an IPython notebook, submit a link to Google Colab or Kaggle.
  - If your solution is a PyTorch project, upload it on GitHub and submit the link to GitHub repo.
- Predictions `csv`: save your model predictions for the `eval` set into `your_university_email.csv` file and upload it to the form. **You want receive the grade if the filename is wrong**.

> [!IMPORTANT]
> Check your solution by running `grading.py` script, described below. It also explains the expected formatting of `your_university_email.csv`. **If the grading script does not work with your `.csv`, your $P$ will be set to zero.**

`grading.py` expects to have a directory `students_solutions` with the following format:

```bash
students_solutions
├── iiivanov.csv # example university username
├── ...
└── pppetrov.csv # example university username
```

To run the script, do:

```bash
# protocol is located next to grading.py
mkdir students_solutions
mv my_username.csv students_solutions/
python3 grading.py
```

We will use the output of this script to grade your solutions.
