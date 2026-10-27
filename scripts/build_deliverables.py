"""Build the academic report and PDF presentation from measured evidence.

Install requirements-artifacts.txt. Run after ml.train, ml.validate and
scripts/check_project.py. Screenshots under tmp/pdfs are optional real captures;
no mock interface or fabricated evidence is substituted when absent.
All generated output stays ignored. Student authorship is explicit in metadata.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "tmp/pdfs"
OUT = ROOT / "output"
BUILD.mkdir(parents=True, exist_ok=True)
(OUT / "pdf").mkdir(parents=True, exist_ok=True)
(OUT / "slides").mkdir(parents=True, exist_ok=True)
B = json.loads((ROOT / "benchmarks/baseline.json").read_text())
L = json.loads((ROOT / "benchmarks/latest.json").read_text())
V = json.loads((ROOT / "benchmarks/validation.json").read_text())
CHECKS = json.loads((ROOT / "benchmarks/checks.json").read_text())
DF = pd.read_csv(ROOT / "dashboard_data.csv")
R = pd.read_csv(ROOT / "benchmarks/regression_predictions.csv")
CURRENT, ORIGINAL = V["current_oof"], V["baseline_oof"]
REG = L["regression"]
NAVY, BLUE, TEAL, ORANGE = "#17324D", "#397FBD", "#258B87", "#BA8547"
INK, MUTED, LINE = "#233244", "#627383", "#DDE5EA"
PALETTE = [BLUE, "#5A9FC6", "#5BAEAA", "#9BBCC6", "#BBCBD5"]
TEAM = "Chirsmriti Srivastava and Lakshya Gupta"
AUTHOR_LINES = "Chirsmriti Srivastava  -  23BBS0093<br/>Lakshya Gupta  -  23BBS0127"
TITLE = "EV Battery Degradation Complaint Mining"
GAIN = (CURRENT["accuracy"] - ORIGINAL["accuracy"]) * 100
MAE_GAIN = (1 - REG["mae"] / B["regression_reconstructed_rf"]["mae"]) * 100

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": LINE,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white"})


def save(fig, name):
    fig.savefig(BUILD / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def charts():
    counts = DF.issue_label.value_counts()
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.barh(counts.index[::-1], counts.values[::-1], color=PALETTE[::-1], height=.6)
    for i, value in enumerate(counts.values[::-1]): ax.text(value + 8, i, str(value), va="center")
    ax.set_xlim(0, 960); ax.set_xlabel("Synthetic records"); ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0); fig.tight_layout(); save(fig, "category_counts.png")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(2); width = .30
    for i, (name, key) in enumerate([("Accuracy", "accuracy"), ("Macro F1", "macro_f1")]):
        values = [ORIGINAL[key] * 100, CURRENT[key] * 100]
        bars = ax.bar(x + (i - .5) * width, values, width, label=name, color=[BLUE, TEAL][i])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5, fontsize=10)
    ax.set_xticks(x, ["Original configuration", "Improved cascade"])
    ax.set_ylim(0, 108); ax.set_ylabel("Score (0-100)"); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "classification_comparison.png")
    cm = np.array(CURRENT["confusion_matrix"])
    labels = ["Capacity", "Charging", "Other", "Range", "Thermal"]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), labels); ax.set_yticks(range(5), labels)
    for i in range(5):
        for j in range(5): ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()*.5 else NAVY)
    ax.set_xlabel("Predicted category"); ax.set_ylabel("Actual category")
    fig.tight_layout(); save(fig, "confusion_matrix.png")
    selection = sorted(L["classification_selection"].items(), key=lambda item: item[1]["mean_macro_f1"])
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [n.replace("_", " ") for n, _ in selection]
    vals = [r["mean_macro_f1"] for _, r in selection]
    ax.barh(names, vals, color=[TEAL if "domain" in n else "#ACC3D2" for n, _ in selection], height=.65)
    for i, value in enumerate(vals): ax.text(value + .012, i, f"{value:.3f}", va="center")
    ax.set_xlim(0, 1.08); ax.set_xlabel("Mean macro F1 in four development folds")
    fig.tight_layout(); save(fig, "candidate_selection.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].scatter(R.actual, R.predicted, s=12, color=TEAL, alpha=.45, edgecolors="none")
    low, high = min(R.actual.min(), R.predicted.min()), max(R.actual.max(), R.predicted.max())
    axes[0].plot([low, high], [low, high], color=BLUE, lw=1, linestyle="--")
    axes[0].set(xlabel="Actual degradation (pp)", ylabel="Predicted degradation (pp)", title="Actual vs. predicted")
    axes[1].scatter(R.predicted, R.residual, s=12, color=BLUE, alpha=.4, edgecolors="none")
    axes[1].axhline(0, color=ORANGE, linestyle="--", lw=1)
    axes[1].set(xlabel="Predicted degradation (pp)", ylabel="Actual minus predicted (pp)", title="Residuals")
    fig.tight_layout(); save(fig, "regression_diagnostics.png")


ST = getSampleStyleSheet()
ST.add(ParagraphStyle(name="BodyAcademic", fontName="Times-Roman", fontSize=11.3, leading=16.2, textColor=colors.HexColor(INK), spaceAfter=11))
ST.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=17, leading=22, textColor=colors.HexColor(NAVY), spaceAfter=20))
ST.add(ParagraphStyle(name="Subsection", fontName="Helvetica-Bold", fontSize=11.5, leading=16, textColor=colors.HexColor(NAVY), spaceBefore=10, spaceAfter=7))
ST.add(ParagraphStyle(name="CaptionAcademic", fontName="Helvetica", fontSize=8.3, leading=12, textColor=colors.HexColor(MUTED), spaceAfter=12))
ST.add(ParagraphStyle(name="CellAcademic", fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor(INK)))
ST.add(ParagraphStyle(name="CoverTitle", fontName="Helvetica-Bold", fontSize=27, leading=35, alignment=TA_CENTER, textColor=colors.HexColor(NAVY)))
ST.add(ParagraphStyle(name="CoverText", fontName="Times-Roman", fontSize=13, leading=21, alignment=TA_CENTER, textColor=colors.HexColor(INK)))
ST.add(ParagraphStyle(name="CodeAcademic", fontName="Courier", fontSize=7.4, leading=11, textColor=colors.HexColor(INK), backColor=colors.HexColor("#F4F7F9"), borderPadding=10, spaceAfter=14))


def p(text): return Paragraph(text, ST["BodyAcademic"])
def h(text): return Paragraph(text, ST["Subsection"])
def cap(text): return Paragraph(text, ST["CaptionAcademic"])
def code(text): return Preformatted(text.strip(), ST["CodeAcademic"])
def image(name, width=480, max_height=370):
    path = BUILD / name
    im = ImageReader(str(path)); w, ht = im.getSize()
    scale = min(width/w, max_height/ht)
    return Image(str(path), width=w*scale, height=ht*scale)

def table(rows, widths=None):
    data = [[Paragraph(html.escape(str(c)), ST["CellAcademic"]) for c in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF1F5")),
       ("LINEBELOW", (0,0), (-1,0), .7, colors.HexColor(LINE)),
       ("LINEBELOW", (0,1), (-1,-1), .35, colors.HexColor(LINE)),
       ("TOPPADDING", (0,0), (-1,-1), 9), ("BOTTOMPADDING", (0,0), (-1,-1), 9),
       ("VALIGN", (0,0), (-1,-1), "TOP")]))
    return t


def footer(c, doc):
    c.setTitle(TITLE + " - Project Report"); c.setAuthor(TEAM)
    w, height = A4
    c.setStrokeColor(colors.HexColor(LINE)); c.setLineWidth(.5)
    if doc.page > 2:
        c.line(54, height-42, w-54, height-42)
        c.setFont("Helvetica", 7); c.setFillColor(colors.HexColor(MUTED))
        c.drawString(54, height-32, "EV BATTERY DEGRADATION COMPLAINT MINING")
    c.line(54, 43, w-54, 43)
    c.setFont("Helvetica", 7.5); c.setFillColor(colors.HexColor(MUTED))
    c.drawString(54, 29, "CBS3009  /  B. Tech [CSBS]  /  2026-27")
    c.drawRightString(w-54, 29, str(doc.page))
    if doc.page <= 2:
        c.setStrokeColor(colors.HexColor(NAVY)); c.setLineWidth(.8)
        c.rect(30, 54, w-60, height-85)


def report():
    story = []; sections = []
    def page(title, items):
        if story: story.append(PageBreak())
        sections.append(title)
        if title: story.append(Paragraph(title, ST["Section"]))
        story.extend(items)
    logo = BUILD / "vit_logo.jpg"
    first = [Spacer(1, 32)]
    if logo.exists(): first += [Image(str(logo), width=170, height=53), Spacer(1, 32)]
    first += [Paragraph("SCHOOL OF COMPUTER SCIENCE AND<br/>ENGINEERING (SCOPE)", ST["CoverText"]), Spacer(1, 65),
        Paragraph("CBS3009<br/>ADVANCED SOCIAL TEXT AND<br/>MEDIA ANALYTICS", ST["CoverTitle"]), Spacer(1,30),
        Paragraph("PROJECT WORK", ST["CoverText"]), Spacer(1,70), Paragraph("B. Tech [CSBS]<br/>Academic Year 2026-27", ST["CoverText"])]
    page("", first)
    page("", [Spacer(1,45), Paragraph("SCHOOL OF COMPUTER SCIENCE AND<br/>ENGINEERING (SCOPE)", ST["CoverText"]), Spacer(1,65),
        Paragraph("EV BATTERY DEGRADATION<br/>COMPLAINT MINING", ST["CoverTitle"]), Spacer(1,20),
        Paragraph("A PROJECT REPORT", ST["CoverText"]), Spacer(1,30), Paragraph("Submitted by", ST["CoverText"]), Spacer(1,15),
        Paragraph(AUTHOR_LINES, ST["CoverText"]), Spacer(1,65), Paragraph("CBS3009 - ADVANCED SOCIAL TEXT AND MEDIA ANALYTICS<br/>B. Tech [CSBS]  /  Academic Year 2026-27", ST["CoverText"])])
    contents = [
        ["SECTION", "TITLE", "PAGE"], ["", "Abstract of the project", "4"], ["", "Acknowledgement", "5"],
        ["1", "Introduction of the project", "6"], ["2", "Literature review", "7-8"], ["3", "Hardware and software requirements", "9"],
        ["4", "Methodology and system design", "10-14"], ["4.1", "Architecture and task separation", "10"],
        ["4.2", "Data description and audit", "11"], ["4.3", "Preprocessing and feature extraction", "12"],
        ["4.4", "Classification, calibration and routing", "13"], ["4.5", "Degradation regression and uncertainty", "14"],
        ["5", "Evaluation and findings", "15-19"], ["5.1", "Validation design and model selection", "15"],
        ["5.2", "Classification comparison and confusion matrix", "16"], ["5.3", "Error analysis and limitations", "17"],
        ["5.4", "Regression results", "18"], ["5.5", "Inference performance", "19"],
        ["6", "Implementation code and output screenshots", "20-22"], ["7", "Design thinking and usability", "23"],
        ["8", "Conclusion and future work", "24"], ["9", "References (APA format)", "25"], ["Appendix", "Presentation and reproducibility evidence", "26"]]
    contents_table = table(contents, [52,375,58])
    contents_table.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    page("CONTENT", [contents_table])
    page("ABSTRACT OF THE PROJECT", [
        p("Electric-vehicle owners describe battery concerns using inconsistent language, including reduced range, slow charging and overheating. This project organises those descriptions into five issue categories, groups recurring wording for exploration, and estimates degradation from five numerical operating inputs. It integrates offline model development, a Flask inference service and a React dashboard designed for students, professors and technicians."),
        p("An audit of the supplied 2,500-row dataset found that every record is synthetic, 1,332 complaint texts are exact duplicates, and all texts reduce to only 25 template families after vehicle names and numbers are normalised. The evaluation therefore keeps complete template families together. The original TF-IDF/SGD configuration is retrained on the same partitions as the improved system so the comparison measures a modelling change rather than a change in the test set."),
        p(f"The improved classifier combines a compact domain-feature logistic model with a richer domain-plus-TF-IDF fallback and sigmoid probability calibration. Across five outer template folds, pooled out-of-fold accuracy improves from {ORIGINAL['accuracy']*100:.2f}% to {CURRENT['accuracy']*100:.2f}%, an increase of {GAIN:.2f} percentage points. Macro F1 improves from {ORIGINAL['macro_f1']:.3f} to {CURRENT['macro_f1']:.3f}. Calibration uses four inner family-separated folds. These are post-selection internal development results, because feature design used this corpus, and are not independent evidence of real-world accuracy."),
        p(f"For degradation prediction, five regression candidates are compared without using battery health, which exactly determines the target in this dataset. Quadratic Ridge is selected. On the fixed 500-row regression test, MAE is {REG['mae']:.3f} percentage points, RMSE is {REG['rmse']:.3f} and R-squared is {REG['r2']:.3f}. This reduces MAE by {MAE_GAIN:.2f}% relative to a reconstructed Random Forest baseline. A separate 400-row calibration set provides a nominal 90% prediction interval, with {REG['interval_observed_coverage']*100:.1f}% observed coverage on the synthetic test."),
        p(f"The application adds batch inference, bounded caching, request validation, uncertainty review flags and a complete degradation endpoint. {CHECKS['tests_passed']} automated tests pass and the frontend builds successfully. A task-based interface supports light and dark modes, plain-language help, persistent form inputs and accessible navigation. The project demonstrates an integrated analytical workflow while identifying the need for diverse, independently labelled complaints and measured battery data."),
        cap("Keywords: electric vehicles, complaint mining, synthetic data, grouped validation, TF-IDF, logistic regression, Ridge regression, probability calibration, usability.")])
    page("ACKNOWLEDGEMENT", [Spacer(1,20),
        p("We express our sincere gratitude to the faculty of the School of Computer Science and Engineering, Vellore Institute of Technology, for their academic guidance and encouragement throughout this project. The coursework in Advanced Social Text and Media Analytics provided the foundation for connecting text representation, machine learning and interpretable visual analysis."),
        p("We thank our teachers and peers for their suggestions on presenting the work clearly and examining the assumptions behind model scores. We also acknowledge the maintainers of Python, pandas, scikit-learn, Flask, React and the other open-source tools used to implement and evaluate the application."),
        p("The project has strengthened our understanding of how data quality, evaluation design and interface decisions influence the usefulness of a machine-learning system. In particular, it emphasised the importance of reproducible comparisons, transparent limitations and explanations that users can understand."),
        Spacer(1,55), Paragraph(AUTHOR_LINES, ST["CoverText"])])
    page("1. INTRODUCTION OF THE PROJECT", [h("1.1 Background"),
        p("Battery condition influences driving range, charging behaviour and the ownership experience of an electric vehicle. Owners also describe perceived changes in forum-style complaints. Such descriptions can help organise recurring concerns, but they do not directly measure electrochemical degradation. A report about reduced winter range, for example, should not automatically become a diagnosis of permanent capacity loss."),
        h("1.2 Problem statement"), p("The project asks how an application can convert complaint text into understandable categories, compare recurring patterns, and offer a separate numerical degradation estimate. It must also show how the models were evaluated and make uncertain or unsupported outputs visible. The supplied prototype lacked a reproducible evaluation workflow and the regression implementation described in its documents."),
        h("1.3 Aim and objectives"), p("The aim is to build and evaluate a usable educational system for EV complaint analysis. The objectives are to audit the dataset; reproduce a comparable baseline; develop and compare text representations and classifiers; add a fast inference path; implement numerical degradation regression without target leakage; expose the results through validated APIs; and present the findings in an accessible dashboard."),
        h("1.4 Scope"), p("The classifier handles Capacity Degradation, Charging Issue, Range/Performance, Thermal Issue and Other. K-Means provides exploratory groups. Regression uses vehicle age, charging cycles, average temperature, fast-charge share and discharge rate. The application operates locally and uses the supplied synthetic collection. It does not collect live social-media posts, forecast a vehicle's remaining life, prove fault causes, or certify battery safety."),
        h("1.5 Expected contribution"), p("The contribution is the complete, inspectable workflow: data audit, documented splits, model selection, calibrated inference, prediction intervals, software checks and a task-based interface. Improvements are quoted against a reconstructed baseline under the same evaluation protocol, with the evidence and its limitations stated together.")])
    page("2. LITERATURE REVIEW", [h("2.1 Battery degradation and complaint evidence"),
        p("Edge et al. (2021) review mechanisms and external stresses associated with lithium-ion battery degradation, including temperature and operating load. This provides motivation for considering operating conditions, but does not validate the synthetic target used here. Complaint text and measured capacity are different sources of evidence and should remain separate in the application."),
        h("2.2 Sparse text representations"), p("Term-frequency/inverse-document-frequency representation weights terms according to their occurrence within documents and the collection. It creates a sparse input suitable for linear models. In this project, unigram and bigram features provide a reproducible baseline. Character n-grams are also tested as a way to capture word fragments and spelling variation. Their inclusion is an experiment rather than an assumed improvement."),
        h("2.3 Linear text classification"), p("Joachims (1998) examines support-vector methods for text categorisation in high-dimensional feature spaces. The original project uses a related linear classifier, SGDClassifier with hinge loss, trained by stochastic optimisation. The present comparison also includes a converged SGD model, LinearSVC, logistic regression and Complement Naive Bayes. These methods suit the small local dataset and permit measured comparisons without requiring a large neural model."),
        h("2.4 Domain knowledge as an explicit feature source"), p("The available collection contains only five text families per class. Generic vocabulary can therefore become tied to a sentence template rather than the underlying concern. This project tests transparent symptom and intent features: thermal language, charging failures, capacity loss, range variation, advice-seeking language and negation. Interactions retain context such as heat during charging. This design is a project hypothesis supported by internal validation, not a published guarantee or proof of transfer to real complaints."),
        cap("Sources: Edge et al. (2021), Joachims (1998), and the scikit-learn 1.6 documentation. Full references appear in Section 9.")])
    page("2. LITERATURE REVIEW (CONTINUED)", [h("2.5 Evaluation with dependent observations"),
        p("Scikit-learn's cross-validation guidance distinguishes training, model selection and final testing. When observations share a subject, source or generator template, related records should stay in the same fold. Here, exact duplicates alone are not enough to identify dependence: changing a vehicle name or a number can create apparently different rows from the same sentence. Grouping at the template-family level is therefore the central evaluation control."),
        h("2.6 Probability calibration"), p("Decision-function margins do not automatically have a probability interpretation. Scikit-learn documents calibration using data that is separate from the estimator's fitted observations. This project uses sigmoid calibration with family-separated out-of-fold predictions. Calibration makes the output interface more interpretable, but an unfamiliar domain can still produce confident errors. Review flags complement the displayed probability."),
        h("2.7 Regression and regularisation"), p("Numerical degradation is a continuous target, so accuracy is not an appropriate metric. Ridge regression penalises large coefficients, while polynomial expansion permits nonlinear terms and interactions. Tree ensembles provide alternative flexible baselines. All transformations must fit within each training fold. The selected model is determined by validation MAE, rather than by a presumption that a more complex model must perform better."),
        h("2.8 Prediction intervals"), p("Angelopoulos and Bates (2021) describe conformal prediction as a method for quantifying predictive uncertainty. In split conformal regression, a fitted predictor is evaluated on a separate calibration set and a residual quantile determines the interval width. The coverage interpretation relies on exchangeability. Real vehicles or inputs outside the synthetic training range do not automatically satisfy that assumption."),
        h("2.9 Implications for this project"), p("The literature motivates sparse baselines, grouped evaluation, calibration and explicit uncertainty. The project combines these ideas in a small educational system. Its strongest defensible claim is improvement within a documented synthetic development benchmark. Independent field validation remains a separate future task.")])
    page("3. HARDWARE AND SOFTWARE REQUIREMENTS", [h("3.1 Hardware and runtime"),
        p("The implemented workflow runs locally on a laptop or desktop with a modern browser. A practical educational setup is 8 GB or more RAM, several gigabytes of free storage for packages and build artifacts, and a Python 3.12 environment. These are suggested development resources, not a measured minimum requirement. Benchmark timings are warm, sequential local measurements with one numerical thread."),
        table([["Layer", "Tools used", "Purpose"], ["Data and ML", "Python 3.12, pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.6.1", "Audit, features, fitting and evaluation"], ["Model storage", "Joblib 1.4.2", "Versioned classifier and regressor artifacts"], ["Service", "Flask 3.1.0, flask-cors 5.0.0", "Validated JSON inference endpoints"], ["Interface", "React 18, Vite 5, Recharts 2", "Interactive task views and data visualisation"], ["Verification", "pytest 8.3.4, Vite build", "API, validation and compilation checks"], ["Documents", "ReportLab, Matplotlib, Poppler", "Report, slide deck, charts and visual review"]], [85,220,180]),
        h("3.2 Local execution"), code("python3.12 -m venv .venv\nsource .venv/bin/activate\npip install -r backend/requirements.txt\npython backend/app.py\n\n# In a second terminal\ncd frontend\nnpm ci\nnpm run dev"),
        p("The browser opens the Vite application on localhost. The backend loads saved models once and does not train during prediction requests. The legacy Streamlit entry point now uses the same prediction service. Optional document-generation packages are separate from the API dependencies.")])
    page("4. METHODOLOGY AND SYSTEM DESIGN", [h("4.1 System architecture"),
        p("The system separates development from inference. Offline scripts audit the CSV, define evaluation partitions, compare candidates and write artifacts and evidence. The Flask service loads those artifacts, validates inputs and returns results. The React interface presents the user's task and the interpretation of the result."),
        table([["Stage", "Input", "Output and responsibility"], ["Audit", "Original CSV and saved models", "Data provenance, template groups, duplicate counts, leakage checks"], ["Offline development", "Recorded training partitions", "Candidate scores, calibrated classifier, numerical regressor"], ["Evaluation", "Family folds and fixed numeric test", "Confusion matrices, predictions, errors, intervals, timings"], ["Inference service", "Complaint text or five numeric values", "Category, confidence, review flag, optional cluster, degradation interval"], ["User interface", "Validated service responses", "Overview, classification, estimate, results and records"]], [95,160,230]),
        h("4.1.1 Three distinct modelling tasks"), p("Classification assigns a known issue category. Clustering groups similar text without making a fault diagnosis. Regression estimates a numerical degradation percentage from operating inputs. A cluster identifier is not a regression result and the complaint classifier is not used as a causal explanation for degradation."),
        h("4.1.2 Artifact and evidence integrity"), p("The dataset and model files have SHA-256 fingerprints. The API displays evaluation evidence only when the saved fingerprints match the current artifacts. Automated check results also carry a source-code fingerprint, so stale test status is not silently shown after code changes. Process-level caches require a server restart when artifacts change."),
        h("4.1.3 Request flow"), p("For a single complaint, the service cleans the text, checks its readability, tries a bounded cache, and then runs the compact model. Lower-confidence rows use the fallback model. Clustering is optional. Batch requests vectorise multiple inputs together. Numerical requests follow a separate strict schema and return a degradation estimate with uncertainty and range warnings.")])
    page("4.2 DATA DESCRIPTION AND AUDIT", [
        table([["Audit item", "Observed value", "Implication"], ["Records", "2,500 synthetic; zero source URLs", "No claim of live or real-world observations"], ["Exact duplicate texts", "1,332", "Row splitting can leak wording"], ["Template families", "25, five per category", "Effective text diversity is much lower than row count"], ["Health and target", "health + degradation = 100 exactly", "Health must not be a regression predictor"], ["Original classifier", "TF-IDF + SGD; max_iter = 4", "Optimisation and wording robustness need evaluation"], ["Original regressor", "Absent from repository", "Historical document metrics cannot be verified as original code results"]], [125,165,195]),
        Spacer(1,18), image("category_counts.png", 480, 220), cap("Figure 1. Class balance in the supplied synthetic dataset. Counts come directly from dashboard_data.csv."),
        p("The class distribution is imbalanced. Capacity Degradation has 829 rows, while Other has 267. Accuracy alone can therefore conceal weak performance on smaller classes. Macro F1 and per-class precision/recall accompany the accuracy score. No timestamps or repeated real-vehicle identifiers are available, so age-bucket charts are cross-sectional summaries rather than time-series evidence.")])
    page("4.3 PREPROCESSING AND FEATURE EXTRACTION", [h("4.3.1 Reproducible text cleaning"),
        p("The service uses the original cleaning transform: lowercase; replace characters outside a-z, digits, period, percent and whitespace with spaces; collapse repeated whitespace; and strip. The same transform is used at prediction time. Readability checks reject punctuation-only or unsupported text. Periods and percentages remain available rather than being discarded."),
        code('def clean_text(text):\n    lowered = str(text).lower()\n    spaced = re.sub(r"[^a-z0-9.%\\s]", " ", lowered)\n    return re.sub(r"\\s+", " ", spaced).strip()'),
        h("4.3.2 Family groups for evaluation"), p("A separate grouping transform substitutes the known vehicle names and numeric strings with placeholders, then hashes the resulting template. Grouping is used to partition records; the family hash is never a model input. Unique families are stratified by category so every fold contains all five categories. Each outer fold holds out one family per category."),
        h("4.3.3 Candidate feature sets"),
        table([["Representation", "Definition", "Role"], ["Word TF-IDF", "Unigrams and bigrams, min_df=2, sublinear term frequency", "Sparse lexical baseline and richer fallback"], ["Character TF-IDF", "Within-word 3-5 character n-grams", "Candidate for spelling and word-fragment variation"], ["Domain features", "Nine binary concepts plus five interactions", "Compact symptom/intent representation"], ["Numeric features", "Five operating inputs; median imputation and scaling", "Leakage-controlled degradation regression"]], [100,250,135]),
        p("Domain features encode thermal language, capacity, range, charging, charging failures, range variation, information-seeking, specific loss and negation. The five interactions retain combinations such as heat during charging and battery-health questions. They are explicit, inspectable design choices and require validation beyond the sentence generator used here.")])
    page("4.4 CLASSIFICATION, CALIBRATION AND ROUTING", [
        p("The development comparison includes converged SGD, word logistic regression, word-plus-character logistic regression, word SVM, word-plus-character SVM, Complement Naive Bayes, domain logistic regression, and domain-plus-word logistic regression. Selection uses mean macro F1 across four development folds. The domain and domain-plus-word models tie in this comparison, so the compact model is used for fast routing and the richer model supplies the fallback."),
        h("4.4.1 Two-stage prediction"),
        code('probabilities = fast_model.predict_proba(cleaned_texts)\nfallback = probabilities.max(axis=1) < 0.90\nif fallback.any():\n    probabilities[fallback] = fallback_model.predict_proba(\n        cleaned_texts[fallback]\n    )\npredicted = classes[probabilities.argmax(axis=1)]'),
        p(f"The routing threshold is 0.90. On the 504-row development holdout, the fast stage handles {L['classification']['fast_path_share']*100:.2f}% of rows without calling the richer model. This coverage is specific to that partition; it is not the full-corpus OOF accuracy or a guarantee about future traffic."),
        h("4.4.2 Calibration and human review"), p("Sigmoid calibration uses family-disjoint out-of-fold scores inside the training partition. The service also returns a review flag when maximum probability is below 0.65, wording has no recognised domain support, or negation requires interpretation. A review flag does not change the suggested category into a verified diagnosis. It asks the user to inspect the result."),
        h("4.4.3 Cache, batch and clustering"), p("Single-record results use a bounded 1,024-entry cache keyed by cleaned text and the clustering option. Returned values are copied so a caller cannot corrupt a cached nested dictionary. Batches contain 1-100 records and perform transformations together. The original clustering model is retained as an exploratory view; assigning five clusters does not mean that each of the five issue labels has its own cluster.")])
    page("4.5 DEGRADATION REGRESSION AND UNCERTAINTY", [h("4.5.1 Inputs and leakage control"),
        p("The five inputs are vehicle age in months, total charging cycles, average temperature in Celsius, fast-charge share from 0 to 1, and average discharge rate in C. Battery health is excluded because it is exactly 100 minus degradation in every record. Text, severity, labels, cluster IDs and previous predictions are also excluded from numerical modelling."),
        h("4.5.2 Partition and model selection"), p("A fixed 80/20 row split with random_state=42 creates a 500-row test and 2,000 development rows. The development portion is split again with seed 43: 1,600 rows for fitting/model selection and 400 for interval calibration. Five-fold validation on the fitting portion compares linear Ridge, quadratic Ridge, Random Forest, Extra Trees and histogram gradient boosting. The best mean MAE selects quadratic Ridge."),
        table([["Candidate", "Five-fold development MAE (pp)"]] + [[name.replace("_", " "), f"{result['mean_mae']:.3f}"] for name, result in L["regression_selection"].items()], [300,185]),
        h("4.5.3 Selected pipeline and interval"), p("Median imputation and standardisation precede degree-two polynomial expansion and Ridge regression with alpha=10. The polynomial terms permit numerical interactions while the penalty controls coefficient magnitude. The selected estimator remains fitted only on the 1,600 fitting rows. Absolute residuals on the separate 400-row calibration set determine a conservative quantile for a nominal 90% split-conformal interval."),
        p(f"The resulting interval half-width is {REG['interval_half_width']:.3f} percentage points. The API clips displayed estimates/interval endpoints to the physical 0-100 range and marks inputs outside any fitted feature range. Clipping supports presentation; it does not validate extrapolation or create a real-world coverage guarantee.")])
    page("5. EVALUATION AND FINDINGS", [h("5.1 Validation design"),
        p("The primary text comparison pools out-of-fold predictions from five outer folds over all 2,500 rows. Every outer fold holds out one complete template family per category, and inner four-fold partitions calibrate probability outputs without using the outer fold's labels for calibration. The same outer partitions evaluate the original configuration and the improved cascade."),
        p("This is post-selection development validation. Domain-feature design used the available corpus, and the 504-row development holdout was examined during iteration. Therefore neither score is presented as an untouched final test or an independent estimate of performance on real complaints. The pooled score is also distinguished from the unweighted mean of fold scores."),
        h("5.1.1 Four-fold candidate comparison"), image("candidate_selection.png", 480, 280),
        cap("Figure 2. Candidate selection on the 1,996-row development partition. Values are mean macro F1 across four template folds, not final test accuracy."),
        p("The unsuccessful generic-only experiment is retained in the evidence files. More iterations, character features or a different generic classifier did not consistently solve the template-generalisation problem. The largest improvement came from changing the representation to express domain concepts and intent. This supports a feature-design explanation for the gain, while leaving real-domain transfer unproven.")])
    page("5.2 CLASSIFICATION RESULTS", [
        table([["Metric", "Original configuration", "Improved cascade"], ["Pooled OOF accuracy", f"{ORIGINAL['accuracy']*100:.2f}%", f"{CURRENT['accuracy']*100:.2f}%"], ["Pooled OOF macro F1", f"{ORIGINAL['macro_f1']:.3f}", f"{CURRENT['macro_f1']:.3f}"], ["Development holdout accuracy", f"{B['reconstructed_sgd_group_test']['accuracy']*100:.2f}%", f"{L['classification']['accuracy']*100:.2f}%"]], [210,138,137]),
        Spacer(1,13), p(f"Across all template folds, accuracy increases by <b>{GAIN:.2f} percentage points</b>. The original configuration correctly classifies {round(ORIGINAL['accuracy']*2500):,} of 2,500 rows; the improved cascade correctly classifies {round(CURRENT['accuracy']*2500):,}. Both are evaluated on the same synthetic template folds."),
        image("confusion_matrix.png", 400, 330), cap("Figure 3. Pooled OOF confusion matrix for the improved cascade. Rows are true labels and columns are predictions. Capacity = Capacity Degradation; Range = Range/Performance."),
        p("Charging and Range/Performance are recovered well in this generator. Remaining confusion involves capacity, thermal wording and general information questions. The next page shows the class-level scores so strong majority-class performance cannot hide those weaknesses.")])
    page("5.3 ERROR ANALYSIS AND VALIDITY LIMITS", [
        table([["Category", "Precision", "Recall", "F1", "Rows"]] + [[label, f"{CURRENT['per_class'][label]['precision']:.3f}", f"{CURRENT['per_class'][label]['recall']:.3f}", f"{CURRENT['per_class'][label]['f1-score']:.3f}", str(int(CURRENT['per_class'][label]['support']))] for label in CURRENT["labels"]], [205,70,70,70,70]),
        h("5.3.1 What the errors reveal"), p("The Other category has the weakest precision. General battery-care questions can resemble capacity complaints, and text mentioning a battery does not establish an actual symptom. Some thermal descriptions also overlap with charging language. These are relevant user-facing failure modes, so the interface displays uncertainty and asks for manual review rather than presenting every label as a fault finding."),
        h("5.3.2 Small effective sample size"), p(f"The 2,500 rows contain only 25 template families. Resampling complete families gives an exploratory 95% bootstrap accuracy interval of {V['family_bootstrap_accuracy_95_interval'][0]*100:.1f}% to {V['family_bootstrap_accuracy_95_interval'][1]*100:.1f}%. Its width reflects limited wording diversity. A row-wise interval treating duplicates as independent would appear much more precise than the evidence warrants."),
        h("5.3.3 Interpretation of confidence"), p("The model uses calibrated probabilities, but no field calibration study has been performed. A short lexical model can miss negation, sarcasm, multilingual text, mixed faults or unfamiliar descriptions. The review threshold is a workflow aid, not a guarantee that accepted predictions are correct. The current system still proposes one category even for a multi-issue complaint."),
        h("5.3.4 Limits of the source data"), p("All source URLs are empty, there are no publication timestamps, and the generator/target construction code was not supplied. The data cannot establish the incidence of faults in any car model, the effect of a charging habit, or a causal degradation mechanism. The model-comparison result remains useful as a reproducible educational experiment within that boundary.")])
    page("5.4 REGRESSION RESULTS", [
        table([["Model / protocol", "MAE (pp)", "RMSE (pp)", "R-squared"], ["Mean predictor, 500-row test", f"{B['regression_mean']['mae']:.3f}", f"{B['regression_mean']['rmse']:.3f}", f"{B['regression_mean']['r2']:.3f}"], ["Reconstructed Random Forest", f"{B['regression_reconstructed_rf']['mae']:.3f}", f"{B['regression_reconstructed_rf']['rmse']:.3f}", f"{B['regression_reconstructed_rf']['r2']:.3f}"], ["Selected quadratic Ridge", f"{REG['mae']:.3f}", f"{REG['rmse']:.3f}", f"{REG['r2']:.3f}"]], [245,80,85,75]),
        Spacer(1,16), p(f"Quadratic Ridge reduces MAE by {MAE_GAIN:.2f}% relative to the reconstructed Random Forest and by {(1-REG['mae']/B['regression_mean']['mae'])*100:.2f}% relative to the mean predictor. It uses fewer fitting rows than the baseline because 400 development rows remain reserved for interval calibration. The modest additional gain is reported without turning R-squared into an accuracy percentage."),
        image("regression_diagnostics.png", 480, 230), cap("Figure 4. Regression predictions and residuals on the fixed 500-row test. The dashed diagonal marks perfect agreement; the residual baseline is zero."),
        p(f"The nominal 90% split-conformal interval covers {REG['interval_observed_coverage']*100:.1f}% of synthetic test targets. Its half-width is {REG['interval_half_width']:.3f} pp, illustrating that meaningful uncertainty remains despite the improved point prediction. The test and calibration partitions are independent of estimator fitting, but they still come from the same synthetic collection."),
        p("The original documents quoted RF MAE=1.120 and R-squared=0.516. Their regressor artifact was absent at the starting commit. The reconstructed RF obtains closely similar values, but the present comparison uses its measured numbers and does not assert an exact reproduction of an unavailable historical model.")])
    timings = V["timings"]
    page("5.5 INFERENCE PERFORMANCE", [
        table([["Warm local workload", "Median ms", "95th percentile ms"], ["Original classification + cluster", f"{B['old_classify_and_cluster_latency']['median_ms']:.4f}", f"{B['old_classify_and_cluster_latency']['p95_ms']:.4f}"], ["Current uncached classification + cluster", f"{timings['current_uncached_classify_and_cluster']['median_ms']:.4f}", f"{timings['current_uncached_classify_and_cluster']['p95_ms']:.4f}"], ["Current classification only", f"{timings['current_classify_only']['median_ms']:.4f}", f"{timings['current_classify_only']['p95_ms']:.4f}"], ["Current repeated cached complaint + cluster", f"{timings['current_cached_classify_and_cluster']['median_ms']:.4f}", f"{timings['current_cached_classify_and_cluster']['p95_ms']:.4f}"], ["Current batch of 100, classification only", f"{timings['batch_100']['median_ms']:.4f}", f"{timings['batch_100']['p95_ms']:.4f}"]], [295,95,95]),
        h("5.5.1 What improved"), p("Batching avoids repeated preprocessing and estimator-call overhead for every item. The bounded cache accelerates repeated identical complaints. Optional clustering avoids paying for a second text transformation when the user only needs a category. Cluster metadata and dataset analytics are cached instead of regrouping the CSV on every request."),
        p(f"For the same classification-plus-clustering work, the repeated cached path is approximately {B['old_classify_and_cluster_latency']['median_ms']/timings['current_cached_classify_and_cluster']['median_ms']:.1f} times faster than the original warm path in these local samples. This is a <b>cache-hit</b> comparison and must not be presented as the speed of a novel complaint. A batch of 100 classification-only records completes in about {timings['batch_100']['median_ms']:.2f} ms, versus approximately {timings['current_classify_only']['median_ms']*100:.2f} ms for 100 sequential current calls."),
        h("5.5.2 Measurement limits"), p("Single-record timings use 100 repetitions after warming; the batch benchmark uses 30. The process limits numerical threading to one. Timings exclude HTTP transport, network latency, browser rendering and cold model loading. The current uncached pipeline remains in the same sub-millisecond range as the original; the project does not claim a dramatic speedup for every new request."),
        h("5.5.3 Model footprint"), p(f"The final classifier bundle is {(ROOT/'final_classifier.joblib').stat().st_size/1024:.1f} KiB on disk. It contains compact and fallback classifiers, calibration parameters and thresholds. This keeps local inference practical, but artifact size alone is not evidence of accuracy or robustness.")])
    screenshots = [p("The interface groups work into five destinations: Overview, Classify a complaint, Estimate degradation, Model results and Browse records. A user can reach a task directly rather than scrolling through a technical dashboard. The screenshots below are captured from the running application and cropped to its content region on an ultrawide monitor.")]
    if (BUILD/"dashboard_overview.png").exists(): screenshots += [image("dashboard_overview.png",480,295), cap("Figure 5. Actual overview screen, cropped to the working interface. The data and chart counts come from the API.")]
    else: screenshots += [p("The overview shows dataset statistics, labelled complaint bars, severity and an age comparison. No screenshot has been substituted for an absent capture.")]
    screenshots += [h("6.1 Shared application services"), p("The Flask and legacy Streamlit entry points now use the same model service. The old prototype attempted to send an already-vectorised matrix into a full text pipeline, then silently retried after an exception. The shared service passes cleaned strings through the correct pipeline directly and exposes validation failures clearly."),
        code('POST /api/predict\n{"text": "Charging is taking much longer than usual."}\n\nPOST /api/predict-batch\n{"texts": ["Battery is very hot.", "Charging is slow."],\n "include_cluster": false}')]
    page("6. IMPLEMENTATION AND OUTPUT SCREENSHOTS", screenshots)
    items = [h("6.2 Complaint and degradation outputs")]
    for name, caption in [("dashboard_classify.png", "Figure 6. Actual complaint-classification result, including the suggested category and calibrated confidence."), ("dashboard_estimate.png", "Figure 7. Actual degradation estimate with its prediction interval and explanatory help.")]:
        if (BUILD/name).exists(): items += [image(name,480,245), cap(caption)]
    if len(items)==1:
        items += [p("Complaint results include the suggested category, calibrated probabilities and a review flag. Degradation results include a bounded estimate, prediction interval and any inputs outside the training range.")]
    items += [p("The backend validates finite numbers, feature names, bounds, text length and batch size. It rejects booleans as numbers, missing or extra regression fields, unreadable complaint strings and oversized request bodies. Stored outputs and battery health cannot be submitted as additional regression predictors.")]
    items += [h("Example numerical input"), table([["Age (months)", "Cycles", "Temperature", "Fast-charge share", "Discharge rate"], ["48", "600", "28 degrees C", "0.30", "1.2 C"]], [90,70,110,120,95])]
    page("6. IMPLEMENTATION OUTPUTS (CONTINUED)", items)
    page("6.3 VERIFICATION AND REPRODUCIBILITY", [
        table([["Check group", "What is verified"], ["Prediction contract", "Class set, probability sum, uncertainty flags and optional cluster"], ["Batch behaviour", "Batch/single agreement, valid array shape and size limit"], ["Regression input validation", "Finite values, physical bounds, unknown/target fields, extrapolation warnings"], ["Artifact integrity", "Disjoint family/fit/calibration/test partitions; stale metric suppression"], ["Cache and analytics", "Returned cache values cannot mutate stored results; all age buckets covered"], ["Frontend build", "React/Vite compiles to a production bundle"]], [150,335]),
        h("6.3.1 Recorded checks"), p(f"The current check receipt records <b>{CHECKS['tests_passed']} passed automated tests</b> and a successful frontend build. Its source fingerprint ties that status to the verified application files. This is a software-check result, not proof of clinical, safety, accessibility or production readiness."),
        h("6.3.2 Reproduction commands"), code("python -m ml.train\npython -m ml.validate\npython scripts/check_project.py\npython scripts/build_deliverables.py"),
        p("The original baseline must be rerun from the starting classifier artifact, available in the repository's initial commit, because the deployed file now stores a bundle. The baseline evidence records all row IDs and family groups. The latest evidence records regression fitting, calibration and test row IDs, classifier candidates, predictions and model hashes."),
        h("6.3.3 Browser and interaction checks"), p("The dashboard is checked in a browser for task navigation, the visible classification and regression responses, and consistency with service outputs. Light and dark appearances are reviewed separately. Screenshots are cropped to exclude the surrounding ultrawide desktop. Automated API tests and build checks remain distinct from the manual visual review.")])
    ui = [h("7.1 Audience and information architecture"), p("The intended user may be a college student learning the project, a professor reviewing evidence, or a technician interpreting a complaint. Five stable destinations match those tasks. The navigation uses words and icons together, while the overview offers direct links to classification and estimation. Forms remain mounted when the user changes views so typed input is preserved. Browser back/forward navigation retains the selected task."),
        h("7.2 Visual hierarchy and palette"), p("The design takes inspiration from Apple's Human Interface Guidelines: clear hierarchy, consistent navigation and readable information. A neutral canvas and white surfaces support system sans-serif typography. Blue marks actions, teal supports measurements and muted orange indicates another evidence type. Dark mode changes the canvas, surfaces, text and chart treatment together. There is no purple or gradient. The implementation uses original interface components rather than Apple branding."),
        h("7.3 Explanation at the point of use"), p("Prediction views explain what to enter, how to read confidence and why a prediction interval exists. Technical details stay in the model-results view or a disclosure. The bottom-left Model insights section gives a compact accuracy, log-loss and test-status summary, linking to the full evidence. Accuracy uses grouped OOF validation; the displayed log loss uses the development holdout and is labelled accordingly."),
        h("7.4 Access and evaluation limits"), p("The interface includes labelled inputs, a skip link, visible focus, headings after navigation, live result announcements and a reduced-motion preference. Small screens replace the sidebar with labelled horizontal navigation. Direct chart labels and legends supplement color. These choices explain the design intent; a formal user study or accessibility conformance audit has not been conducted.")]
    page("7. DESIGN THINKING AND USABILITY", ui)
    page("8. CONCLUSION AND FUTURE WORK", [h("8.1 Conclusion"),
        p(f"The improved project connects model development, calibrated prediction, numerical uncertainty and a usable interface. Pooled template OOF classification accuracy rises from {ORIGINAL['accuracy']*100:.2f}% to {CURRENT['accuracy']*100:.2f}%, with macro F1 rising from {ORIGINAL['macro_f1']:.3f} to {CURRENT['macro_f1']:.3f}. The {GAIN:.2f}-point accuracy gain is measured using the same family partitions for the original and improved pipelines."),
        p(f"Degradation regression now exists in the code and interface. Quadratic Ridge achieves MAE={REG['mae']:.3f} pp and R-squared={REG['r2']:.3f} on a fixed 500-row synthetic test. This improves on the reconstructed Random Forest MAE by {MAE_GAIN:.2f}%, while reserving calibration data for a prediction interval. Batched and cached inference improve repeated workloads, and {CHECKS['tests_passed']} automated tests document the service's expected behaviour."),
        h("8.2 What the project demonstrates"), p("The largest lesson is that evaluation design changes how a model score should be understood. Thousands of generated rows did not provide thousands of independent examples of language. Grouping templates exposed the original model's weakness, and transparent domain features improved internal performance. Keeping the limitations alongside the result makes the educational contribution more defensible."),
        h("8.3 Prioritised future work"), p("The first priority is to obtain diverse, permissioned real complaints, define annotation rules, and have multiple annotators label them before a frozen external evaluation. Real vehicle or thread IDs should determine groups, and chronological splits should be used when dates become available. Negation, multi-issue complaints, spelling errors and multilingual descriptions need explicit challenge sets."),
        p("For regression, the next step is measured battery-health data with documented target construction, operating history and repeated vehicle observations. Only then should causal interpretations, more complex neural models or deployment decisions be considered. Interface evaluation should observe students, professors and technicians completing concrete tasks, including reading uncertainty and distinguishing synthetic examples from real evidence.")])
    refs = [
        "Angelopoulos, A. N., &amp; Bates, S. (2021). <i>A gentle introduction to conformal prediction and distribution-free uncertainty quantification</i>. arXiv. https://doi.org/10.48550/arXiv.2107.07511",
        "Apple. (n.d.). <i>Sidebars</i>. Human Interface Guidelines. Retrieved October 27, 2026, from https://developer.apple.com/design/human-interface-guidelines/sidebars",
        "Apple. (n.d.). <i>Navigation and search</i>. Human Interface Guidelines. Retrieved October 27, 2026, from https://developer.apple.com/design/human-interface-guidelines/navigation-and-search",
        "Edge, J. S., O'Kane, S., Prosser, R., Kirkaldy, N. D., Patel, A. N., Hales, A., Ghosh, A., Ai, W., Campbell, I. D., Marzook, M., Radhakrishnan, G. J., Wang, K., Patel, Y., Wu, B., &amp; Offer, G. J. (2021). Lithium ion battery degradation: What you need to know. <i>Physical Chemistry Chemical Physics, 23</i>, 8200-8221. https://doi.org/10.1039/D1CP00359C",
        "Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. In <i>Machine Learning: ECML-98</i> (pp. 137-142). Springer. https://doi.org/10.1007/BFb0026683",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &amp; Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html",
        "Scikit-learn developers. (2025). <i>Cross-validation: Evaluating estimator performance</i> (Version 1.6.1). https://scikit-learn.org/1.6/modules/cross_validation.html",
        "Scikit-learn developers. (2025). <i>Probability calibration</i> (Version 1.6.1). https://scikit-learn.org/1.6/modules/calibration.html",
    ]
    page("9. REFERENCES (APA FORMAT)", [p(ref) for ref in refs])
    page("APPENDIX: PRESENTATION AND EVIDENCE", [h("A. Presentation"),
        p("The accompanying PDF slide deck summarises the problem, audit, methodology, measured improvements, inference behaviour, regression findings, interface design and limitations. It preserves the same student names and registration numbers as the report. The deck is generated from the same benchmark files as this document."),
        h("B. Evidence map"), table([["Repository evidence", "Contents"], ["benchmarks/baseline.json", "Original artifact audit, reconstructed baseline, row/family partitions"], ["benchmarks/latest.json", "Selection results, development holdout, regression test, hashes"], ["benchmarks/validation.json", "Five outer-fold comparison, inner calibration, family uncertainty, timings"], ["benchmarks/*predictions.csv", "Individual predictions for error inspection"], ["benchmarks/checks.json", "Verified test/build status and source fingerprint"], ["docs/DESIGN.md", "Design rationale, usability decisions and inspiration sources"]], [235,250]),
        h("C. Provenance"), p(f"Dataset SHA-256: <font name='Courier' size='8'>{B['dataset_sha256']}</font>. The original source artifacts correspond to starting commit e69fa94. Model and evaluation files retain their own hashes. Generated documents and screenshots are excluded from Git; the source code and reproducible evidence remain versioned."),
        h("D. Video"), p("No video recording was supplied or created for this revision. The project does not include an invented video link. The local dashboard and slide deck support a live demonstration of the implemented workflow.")])
    path = OUT / "pdf/EV_Battery_Degradation_Final_Report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=54, leftMargin=54, topMargin=65, bottomMargin=60, title=TITLE, author=TEAM)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def slide_text(c, text, x, top, width, size=20, color=INK, bold=False, leading=None):
    style = ParagraphStyle("slide", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size,
                           leading=leading or size*1.38, textColor=colors.HexColor(color))
    para = Paragraph(text, style); _, height = para.wrap(width, 700)
    para.drawOn(c, x, 720-top-height)
    return height


def slide_image(c, name, x, top, width, height):
    path = BUILD / name
    im = ImageReader(str(path)); w,h = im.getSize(); s=min(width/w,height/h)
    c.drawImage(im, x+(width-w*s)/2, 720-top-h*s, width=w*s, height=h*s, mask="auto")


def deck():
    path = OUT / "slides/EV_Battery_Degradation_Final_Presentation.pdf"
    c = canvas.Canvas(str(path), pagesize=(1280,720))
    c.setTitle(TITLE + " - Final Presentation"); c.setAuthor(TEAM)
    number = 0
    def base(title, source="Project benchmark artifacts; synthetic data only", title_size=34):
        nonlocal number
        if number: c.showPage()
        number += 1
        c.setFillColor(colors.white); c.rect(0,0,1280,720,fill=1,stroke=0)
        slide_text(c,title,68,54,1144,title_size,NAVY,True)
        slide_text(c,source,68,671,1070,11,MUTED)
        slide_text(c,str(number),1190,671,30,11,MUTED)
    def bullets(items, x=72, top=160, width=1100, size=24, gap=28):
        for text in items: top += slide_text(c,text,x,top,width,size) + gap
    base("EV Battery Degradation<br/>Complaint Mining", "CBS3009  /  Advanced Social Text and Media Analytics  /  Academic Year 2026-27", title_size=52)
    slide_text(c,"Classification, validation and an improved user experience",72,232,1030,26,MUTED)
    slide_text(c,AUTHOR_LINES,72,438,800,22,NAVY)
    slide_text(c,"Final project presentation",72,354,800,18,TEAL,True)
    base("Project scope")
    bullets(["Classify an owner’s complaint into one of five issue categories.", "Explore recurring wording and inspect the source records.", "Estimate numerical degradation from operating inputs, with uncertainty.", "Give students, professors and technicians a clear task-based interface."],size=25,gap=26)
    base("The audit changed the evaluation")
    for value,label,x in [("2,500","synthetic rows",72),("1,332","duplicate texts",478),("25","template families",875)]:
        slide_text(c,value,x,180,350,64,TEAL,True); slide_text(c,label,x,270,350,23,MUTED)
    slide_text(c,"Changing a vehicle name or number does not create a new language example. Entire template families stay together in validation.",72,389,1100,27,NAVY)
    slide_text(c,"Battery health + degradation = 100 exactly. Health is excluded from regression inputs.",72,515,1100,23,MUTED)
    base("System design")
    bullets(["Offline development: audit data, define partitions, compare models and record evidence.", "Flask service: validate inputs, load models once and return predictions.", "React interface: Overview, Classify, Estimate, Model results and Browse records.", "Separate tasks: a complaint category, a text cluster and a degradation estimate answer different questions."],size=23,gap=24)
    base("Validation protocol")
    bullets(["Five outer folds each hold one unseen template family from every category.", "Four inner family folds calibrate probability outputs.", "The same outer partitions compare the original configuration with the improved cascade.", "Feature design used this corpus. Results are internal development validation, with no independent real-world test."],size=24,gap=24)
    base("Domain features outperform generic text variants")
    slide_image(c,"candidate_selection.png",70,147,1130,450)
    slide_text(c,"Mean macro F1 on four development folds. Candidate selection is separate from the reported pooled OOF comparison.",72,602,1100,17,MUTED)
    base(f"Accuracy improves by {GAIN:.2f} percentage points")
    slide_text(c,f"{ORIGINAL['accuracy']*100:.2f}%",72,170,450,66,MUTED,True)
    slide_text(c,"Original configuration",72,264,440,22,MUTED)
    slide_text(c,f"{CURRENT['accuracy']*100:.2f}%",650,170,450,66,TEAL,True)
    slide_text(c,"Improved cascade",650,264,440,22,NAVY)
    slide_text(c,f"Macro F1: {ORIGINAL['macro_f1']:.3f} to {CURRENT['macro_f1']:.3f}",72,380,1080,30,NAVY,True)
    slide_text(c,"Pooled out-of-fold results over 2,500 synthetic rows. Five whole-template outer folds with inner probability calibration.",72,466,1080,23,MUTED)
    base("Where the classifier still makes mistakes")
    slide_image(c,"confusion_matrix.png",70,143,650,466)
    slide_text(c,"Other is the weakest category",780,172,410,27,NAVY,True)
    slide_text(c,"General battery-care questions can resemble capacity complaints. Heat descriptions also overlap with charging language.",780,261,410,22,MUTED)
    slide_text(c,"Review flags expose uncertainty, unsupported wording and negation.",780,448,410,22,TEAL)
    base("A compact model handles the confident cases")
    bullets([f"The fast stage handles {L['classification']['fast_path_share']*100:.2f}% of the 504-row development holdout at a 0.90 confidence threshold.", "The richer domain-plus-word model handles the remaining rows.", "Batch prediction shares transformation overhead. A bounded cache accelerates repeated identical complaints.", "Clustering is optional, so a classification-only task avoids unnecessary work."],size=24,gap=22)
    base("Performance improvements depend on the workload")
    t = V["timings"]
    rows=[("Original classification + cluster",B['old_classify_and_cluster_latency']['median_ms']),
          ("Current uncached classification + cluster",t['current_uncached_classify_and_cluster']['median_ms']),
          ("Current cache-hit classification + cluster",t['current_cached_classify_and_cluster']['median_ms']),
          ("Current batch of 100, classification only",t['batch_100']['median_ms'])]
    for i,(label,value) in enumerate(rows):
        slide_text(c,label,72,165+i*91,850,24,NAVY)
        slide_text(c,f"{value:.4f} ms",947,165+i*91,260,25,TEAL,True)
    slide_text(c,"Warm local measurements, excluding HTTP and cold loading. A cache hit is not the cost of a new complaint.",72,580,1100,20,MUTED)
    base("Degradation regression")
    bullets(["Five inputs: vehicle age, charging cycles, temperature, fast-charge share and discharge rate.", "Health, issue labels, severity and stored predictions are excluded to prevent leakage.", "Quadratic Ridge wins five-fold MAE selection against four alternatives.", "1,600 fitting rows, 400 interval-calibration rows and a fixed 500-row synthetic test."],size=24,gap=24)
    base("Regression improves modestly, with uncertainty")
    slide_text(c,f"{REG['mae']:.3f} pp",72,154,440,54,TEAL,True)
    slide_text(c,f"MAE, {MAE_GAIN:.2f}% lower than reconstructed RF",72,233,1100,23,MUTED)
    slide_image(c,"regression_diagnostics.png",65,306,1145,270)
    slide_text(c,f"R-squared {REG['r2']:.3f}. Nominal 90% interval covers {REG['interval_observed_coverage']*100:.1f}% of synthetic test targets. R-squared is not accuracy.",72,604,1110,18,MUTED)
    base("An interface built around the user’s task")
    if (BUILD/"dashboard_overview.png").exists(): slide_image(c,"dashboard_overview.png",65,139,1148,460)
    slide_text(c,"Persistent navigation, plain-language guidance, readable charts and a compact model-insights summary.",72,614,1100,18,MUTED)
    base("Light and dark mode, with the same evidence")
    if (BUILD/"dashboard_dark.png").exists(): slide_image(c,"dashboard_dark.png",65,137,1148,454)
    else: bullets(["System appearance by default, with a saved light/dark preference.", "Graphite neutrals, blue actions, teal evidence and muted orange.", "No purple or gradients. Text and labels carry meaning alongside color."],size=26)
    slide_text(c,"Design rationale and usability comments are documented in the implementation and project report.",72,610,1100,18,MUTED)
    base("Validation and software checks")
    slide_text(c,f"{CHECKS['tests_passed']} tests passed",72,171,1080,55,TEAL,True)
    bullets(["Probability outputs, batch parity, input validation and prediction intervals.", "Disjoint partitions, cache integrity and stale-metric suppression.", "Frontend production build and browser task checks.", "Recorded status is tied to source fingerprints."],top=282,size=23,gap=19)
    base("Limitations and the next study")
    bullets(["Only 25 generated template families. The 95% family-bootstrap accuracy interval is wide.", "No independent real-world test, no causal battery diagnosis and no live social-media collection.", "Next: diverse permissioned complaints, independent annotation and a frozen external test.", "Measured vehicle histories and observed task completion should guide the next modelling and usability study."],size=24,gap=22)
    base("Conclusion")
    slide_text(c,f"{CURRENT['accuracy']*100:.2f}% grouped development accuracy",72,174,1120,40,TEAL,True)
    slide_text(c,f"{GAIN:.2f} percentage points above the original configuration",72,246,1110,27,NAVY)
    slide_text(c,"A reproducible comparison, complete prediction workflows and a clearer interface.",72,349,1100,28,MUTED)
    slide_text(c,AUTHOR_LINES,72,495,1100,21,NAVY)
    c.save(); return path, number


if __name__ == "__main__":
    charts()
    report_path = report()
    deck_path, slide_count = deck()
    print(json.dumps({"report": str(report_path), "presentation": str(deck_path), "slides": slide_count}, indent=2))
