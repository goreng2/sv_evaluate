import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import det_curve, DetCurveDisplay
import numpy as np

sns.set_theme()


def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [float(line.rstrip()) for line in f.readlines()]

    return lines


def create_df(dataset, class_name):
    df = pd.DataFrame({
        "class": [class_name for i in dataset],
        "score": dataset
    })

    return df


def concat_df(*df):
    return pd.concat(list(df), ignore_index=True)


def EER(positive_scores, negative_scores):
    """
    https://github.com/speechbrain/speechbrain/blob/fe16ffc9911cb41fecbd8d1160719f5f04ddfd2e/speechbrain/utils/metric_stats.py#L456

    Computes the EER (and its threshold).
    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_eer, threshold = EER(positive_scores, negative_scores)
    >>> val_eer
    0.0
    """
    # Convert data type
    if type(positive_scores) is not torch.Tensor or \
     type(negative_scores) is not torch.Tensor:
        positive_scores = torch.tensor(positive_scores)
        negative_scores = torch.tensor(negative_scores)

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Variable to store the min FRR, min FAR and their corresponding index
    min_index = 0
    final_FRR = 0
    final_FAR = 0

    for i, cur_thresh in enumerate(thresholds):
        pos_scores_threshold = positive_scores <= cur_thresh
        FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[0]
        del pos_scores_threshold

        neg_scores_threshold = negative_scores > cur_thresh
        FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[0]
        del neg_scores_threshold

        # Finding the threshold for EER
        if (FAR - FRR).abs().item() < abs(final_FAR - final_FRR) or i == 0:
            min_index = i
            final_FRR = FRR.item()
            final_FAR = FAR.item()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    # print(f"final_FAR {final_FAR}, final_FRR {final_FRR}")
    EER = (final_FAR + final_FRR) / 2

    return float(EER), float(thresholds[min_index])


def ppndf(p):
    """
    https://gitlab.idiap.ch/bob/bob.measure/-/blob/master/src/bob/measure/_library.py#L1344

    Returns the Deviate Scale equivalent of a false rejection/acceptance ratio

    The algorithm that calculates the deviate scale is based on function
    ``ppndf()`` from the NIST package DETware version 2.1, freely available on
    the internet. Please consult it for more details. By 20.04.2011, you could
    find such package `here <http://www.itl.nist.gov/iad/mig/tools/>`_.

    The input to this function is a cumulative probability.  The output from
    this function is the Normal deviate that corresponds to that probability.
    For example:

    -------+--------
     INPUT | OUTPUT
    -------+--------
     0.001 | -3.090
     0.01  | -2.326
     0.1   | -1.282
     0.5   |  0.0
     0.9   |  1.282
     0.99  |  2.326
     0.999 |  3.090
    -------+--------


    Parameters
    ==========

    p : numpy.ndarray (2D, float)

        The value (usually FPR or FNR) for which the PPNDF should be calculated


    Returns
    =======

    ppndf : numpy.ndarray (2D, float)

        The derivative scale of the given value

    """

    # threshold
    epsilon = np.finfo(np.float64).eps
    p_new = np.copy(p)
    p_new = np.where(p_new >= 1.0, 1.0 - epsilon, p_new)
    p_new = np.where(p_new <= 0.0, epsilon, p_new)

    q = p_new - 0.5
    abs_q_smaller = np.abs(q) <= 0.42
    abs_q_bigger = ~abs_q_smaller

    retval = np.zeros_like(p_new)

    # first part q<=0.42
    q1 = q[abs_q_smaller]
    r = np.square(q1)
    opt1 = (
        q1
        * (
            ((-25.4410604963 * r + 41.3911977353) * r + -18.6150006252) * r
            + 2.5066282388
        )
        / (
            (
                ((3.1308290983 * r + -21.0622410182) * r + 23.0833674374) * r
                + -8.4735109309
            )
            * r
            + 1.0
        )
    )
    retval[abs_q_smaller] = opt1

    # second part q>0.42
    # r = sqrt (log (0.5 - abs(q)));
    q2 = q[abs_q_bigger]
    r = p_new[abs_q_bigger]
    r[q2 > 0] = 1 - r[q2 > 0]
    if (r <= 0).any():
        raise RuntimeError("measure::ppndf(): r <= 0.0!")

    r = np.sqrt(-1 * np.log(r))
    opt2 = (
        ((2.3212127685 * r + 4.8501412713) * r + -2.2979647913) * r
        + -2.7871893113
    ) / ((1.6370678189 * r + 3.5438892476) * r + 1.0)
    opt2[q2 < 0] *= -1
    retval[abs_q_bigger] = opt2

    return retval


def draw_distribution(total_df, threshold):
    # Draw
    sns.displot(
        data=total_df,
        kind="kde",
        x="score",
        hue="class",
        fill=True,
    )

    plt.axvline(
        x=threshold,
        linestyle="--",
        color="mediumseagreen",
        label=f"Threshold {round(threshold, 2)}",
    )

    plt.title("Positive & Negative Distribution")
    plt.legend()

    # Save
    plt.savefig(fname="Positive & Negative Distribution", bbox_inches="tight")


def draw_DETcurves(total_df, EER_value):
    # Preprocess
    y_true = total_df["class"].transform(lambda x: 1 if x == "positive" else 0).to_numpy()
    y_score = total_df["score"].to_numpy()
    fpr, fnr, thresholds = det_curve(y_true, y_score)

    # Draw
    display = DetCurveDisplay(
        fpr=fpr,
        fnr=fnr,
        estimator_name="DET Curve"
    )
    display.plot()

    plt.scatter(
        x=ppndf(EER_value),
        y=ppndf(EER_value),
        c="#d62728",  # Color Red,
        label=f"EER: {round(EER_value * 100, 2)}%",
    )

    plt.title("Detection Error Tradeoff (DET) curves")
    plt.legend()

    # Save
    plt.savefig(fname="Detection Error Tradeoff (DET) curves")


def main():
    # 1. Load Data
    pos = load(path="target.TD_Small.NoSuf")
    neg = load(path="nontarget.TD_Small.NoSuf")

    # 2. Create DataFrame
    pos_df = create_df(dataset=pos, class_name="positive")
    neg_df = create_df(dataset=neg, class_name="negative")
    total_df = concat_df(pos_df, neg_df)

    # 3. Calc EER, Threshold
    EER_value, threshold = EER(positive_scores=pos, negative_scores=neg)

    # 4. Draw & Save Graph
    draw_distribution(total_df, threshold)
    draw_DETcurves(total_df, EER_value)


if __name__ == '__main__':
    main()
