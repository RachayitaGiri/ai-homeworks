# CECS 551 HW 2

In this assignment you will practice putting together a simple image classification pipeline, based on the **k-Nearest Neighbor** or the **SVM/Softmax classifier**. The goals of this assignment are as follows:

- understand the basic **Image Classification pipeline** and the data-driven approach (train/predict stages)
- understand the train/val/test **splits** and the use of validation data for **hyperparameter tuning**.
- develop proficiency in writing **efficient vectorized** code with numpy
- implement and apply a k-Nearest Neighbor (**kNN**) classifier
- implement and apply a Multiclass Support Vector Machine (**SVM**) classifier
- implement and apply a **Softmax** classifier
- implement and apply a **Two layer neural network** classifier
- understand the differences and tradeoffs between these classifiers
- get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)


## Submitting your work

Submit **pdf** of the completed iPython notebooks to **BeachBoard**.

To produce a pdf of your work, you can first convert each of the .ipynb files to HTML. To do this, simply run

```bash
ipython nbconvert --to html FILE.ipynb
```

for each of the notebooks, where ``FILE.ipynb`` is the notebook you want to convert. Then you can convert the HTML files to PDFs with your favorite web browser.

**Important: Please make sure that the submitted notebooks have been run and the cell outputs are visible.**