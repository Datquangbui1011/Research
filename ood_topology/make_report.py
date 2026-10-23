"""Build the 2-page PDF report (OOD_Topology_Report.pdf) from the 3-seed results."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "OOD_Topology_Report.pdf"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_DIR / "Arial Bold.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONT_DIR / "Arial Italic.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Arial", fontSize=8.4, leading=10.1, spaceAfter=2)
small = ParagraphStyle("small", parent=body, fontSize=7.4, leading=8.8)
cell = ParagraphStyle("cell", parent=body, fontSize=7.4, leading=8.6, spaceAfter=0)
h1 = ParagraphStyle("h1", parent=body, fontName="Arial-Bold", fontSize=14, leading=17, alignment=TA_CENTER, spaceAfter=2)
sub = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, fontSize=8.6, textColor=colors.HexColor("#444444"), spaceAfter=6)
h2 = ParagraphStyle("h2", parent=body, fontName="Arial-Bold", fontSize=10, leading=12, spaceBefore=4, spaceAfter=1.5,
                    textColor=colors.HexColor("#1f3b63"))
B = "<b>{}</b>".format


def table(rows, widths, bold_rows=(), header_bg="#1f3b63"):
    hdr = ParagraphStyle("hdr", parent=cell, textColor=colors.white, fontName="Arial-Bold")
    data = [[Paragraph(str(c), hdr if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b7c3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
    ]
    for r in bold_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#dfeaf7")))
    t.setStyle(TableStyle(style))
    return t


def box(paragraphs, width, bg="#eef3fa", border="#1f3b63"):
    t = Table([[paragraphs]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(border)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def verdict(text, color):
    return f'<font color="{color}"><b>{text}</b></font>'


GOOD, OK, BAD, BASE = "#1b7a3a", "#a86a00", "#b3261e", "#555555"


def build():
    W = letter[0] - 1.3 * inch
    story = []
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=0, spaceAfter=2)

    def bullets(items):
        for it in items:
            story.append(Paragraph(it, bullet, bulletText="•"))

    # ------------------------------------------------------------------ title + at a glance
    story.append(Paragraph("Topology of Neural-Network Layers for Out-of-Distribution Detection", h1))
    story.append(Paragraph("Dat Bui &nbsp;·&nbsp; October 2026 &nbsp;·&nbsp; code and full results: "
                           "github.com/Datquangbui1011/Research", sub))
    glance = ParagraphStyle("glance", parent=body, spaceAfter=1.5)
    story.append(box([
        Paragraph(B("At a glance"), ParagraphStyle("gh", parent=h2, spaceBefore=0)),
        Paragraph(B("Task: ") + "find a model used in the six papers of the literature report, implement all six "
                  "approaches on it, use them to detect out-of-distribution (OOD) inputs, and compare how well each works.", glance),
        Paragraph(B("Model: ") + "the Bayesian MLP 784→64→16→10 from the pTU paper [3], trained on MNIST "
                  "(97.2 % test accuracy, close to the paper's 97.4 %).", glance),
        Paragraph(B("Test: ") + "can each method tell MNIST digits apart from 7 kinds of unfamiliar images? "
                  "Every result is averaged over 3 independently trained models.", glance),
        Paragraph(B("Main result: ") + "5 of the 6 topological approaches beat the standard confidence baseline. "
                  "The best one, [5], reaches 96.5 / 100 AUROC versus 74.1 for confidence. "
                  "Neural Persistence [1] cannot detect OOD inputs at all.", glance),
    ], W))

    # ------------------------------------------------------------------ 1. model
    story.append(Paragraph("1. The shared model", h2))
    bullets([
        B("Why this model: ") + "papers [1], [2] and [3] all study small fully connected networks (MLPs) on MNIST, and "
        "[3] gives the exact architecture, prior and training rule, so the setup can be reproduced.",
        B("What it is: ") + "784 inputs (28×28 pixels) → 64 → 16 → 10 classes, ReLU, trained as a Bayesian network "
        "(Bayes-by-Backprop), so each weight has a distribution, not a single value.",
        B("How it is used: ") + "methods [1], [2], [4], [5], [6] use the average weights; [3] (pTU) uses 30 random "
        "weight samples, as in the paper.",
        B("One change from the paper: ") + "the KL term of the loss is scaled by 0.1. With the full term, accuracy stalls "
        "at 94.5 %, and [3] does not say how it weighted this term.",
    ])

    # ------------------------------------------------------------------ 2. approaches
    story.append(Paragraph("2. The six approaches", h2))
    story.append(Paragraph("All six turn a layer of the network into a shape (a graph or a cloud of points), track how "
                           "that shape connects up as a threshold changes (persistent homology), and summarize it with "
                           "numbers. An input whose numbers look unusual compared with training data is flagged as OOD.", body))
    rows = [
        ["Paper", "Main idea in one sentence", "OOD score used here"],
        ["[1] Neural Persistence<br/>Rieck et al. 2019", "How strongly a few weights stand out in each layer of the trained network (uses weights only, no input).",
         "Same number computed on each input's activations (my adaptation, since the original ignores the input)."],
        ["[2] Topological Uncertainty<br/>Lacombe et al. 2021", "Each input activates the network in a pattern; compare that pattern's topology with typical training inputs.",
         "Distance to the average training pattern of the predicted class."],
        ["[3] pTU<br/>Yeh &amp; Yang 2025", "Run the input through 30 sampled versions of a Bayesian network; if their topologies disagree, the input is uncertain.",
         "Spread (variance) of the 30 topologies; plus the paper's statistical test on groups of images."],
        ["[4] Persistent Laplacian<br/>Wolf et al. 2025", "Eigenvalues of a matrix built from the shape: zero eigenvalues = topology, nonzero ones = extra geometry.",
         "Number of zero eigenvalues (β<sub>0</sub>) and the nonzero eigenvalues at 3 thresholds."],
        ["[5] Data manifold<br/>Magai &amp; Ayzenberg 2022", "Treat all images at one layer as a point cloud and measure its loops, clusters and dimension.",
         "Groups of images: cloud topology + dimension. Single image (my adaptation): how much it changes its 20 nearest training neighbours' topology."],
        ["[6] CNN activations<br/>Purvine et al. 2023", "Compare the topology of different layers' point clouds with a distance between diagrams.",
         "Groups of images: distance from the group's diagrams to MNIST's diagrams."],
    ]
    story.append(table(rows, [1.45 * inch, 2.9 * inch, W - 4.35 * inch]))

    # ------------------------------------------------------------------ 3. test
    story.append(Paragraph("3. How the methods were tested", h2))
    bullets([
        B("Normal data: ") + "2000 MNIST test digits. " + B("OOD data") + " (2000 each, all shown as 28×28 grayscale): "
        "Fashion-MNIST (clothes), KMNIST (Japanese characters), CIFAR-10 (photos), SVHN (house numbers), "
        "Gaussian noise, uniform noise, and MNIST-flip (digits with 100 pixels inverted).",
        B("Score: ") + "AUROC = how well a method ranks OOD images above normal ones. "
        + B("50 = random guessing, 100 = perfect.") + " FPR95 (lower is better) = % of OOD images missed when 95 % of normal "
        "images are accepted.",
        B("Two settings: ") + B("one image at a time") + " (the usual OOD task) and " + B("groups of images")
        + " (is this whole batch from a different source?), which is how papers [3], [5], [6] use topology.",
        B("Baselines: ") + "softmax confidence (MSP, the baseline in [2]), Bayesian mutual information, and distance to "
        "the nearest training images (k-NN).",
    ])

    # ------------------------------------------------------------------ 4. results
    sec4 = [Paragraph("4. Results", h2)]
    sec4.append(Paragraph(B("4a. One image at a time") + " — average over the 7 OOD sets, mean ± std over 3 trained models, "
                           "ranked from best to worst. Grey rows are baselines.", body))
    rows = [
        ["#", "Method", "AUROC ↑", "FPR95 ↓", "Verdict"],
        ["1", "[5] Data manifold, single-image version", B("96.5 ± 0.4"), B("16.9 ± 2.3"),
         verdict("Best", GOOD) + " — but it mainly measures distance to the training data"],
        ["2", "k-NN distance (baseline)", "88.4 ± 0.9", "46.4 ± 1.9", verdict("Strong baseline", BASE)],
        ["3", "[2] TU with layers rescaled (my fix)", "87.4 ± 0.6", "60.1 ± 1.6",
         verdict("Good", GOOD) + " — +8 points over the paper's version"],
        ["4", "[4] Persistent Laplacian, zero eigenvalues only (β<sub>0</sub>)", "85.8 ± 0.1", "50.3 ± 1.6",
         verdict("Good", GOOD) + " — very stable"],
        ["5", "[3] pTU", "83.0 ± 5.2", "39.4 ± 7.0", verdict("Good but unstable", OK) + " — misses Gaussian noise"],
        ["6", "[4] Persistent Laplacian, full spectrum", "82.6 ± 0.8", "59.3 ± 1.5",
         verdict("OK", OK) + " — extra eigenvalues make it worse"],
        ["7", "Bayesian mutual information (baseline)", "81.4 ± 4.8", "60.2 ± 9.0", verdict("Baseline", BASE)],
        ["8", "[2] TU (paper version)", "79.1 ± 0.8", "73.1 ± 3.1", verdict("OK", OK) + " — beats confidence"],
        ["9", "Softmax confidence, MSP (baseline)", "74.1 ± 4.8", "74.1 ± 3.0",
         verdict("Weak baseline", BASE) + " — fooled by confident mistakes"],
        ["10", "[1] Neural Persistence", "55.4 ± 5.0", "89.2 ± 4.9", verdict("Fails", BAD) + " — near random guessing"],
    ]
    t = table(rows, [0.25 * inch, 2.75 * inch, 0.75 * inch, 0.75 * inch, W - 4.5 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e6e6e6")) for r in (2, 7, 9)]))
    sec4.append(t)
    story.append(KeepTogether(sec4))
    story.append(Spacer(1, 4))
    story.append(Image(str(ROOT / "results" / "summary_seeds.png"), width=0.74 * W, height=0.74 * W * 750 / 2100))
    story.append(Paragraph("Figure 1. Average AUROC (mean ± std over 3 models). Grey = baselines, blue = topological "
                           "approaches; † = evaluated on 1000 images per set. Dotted line = random guessing.", small))

    story.append(Spacer(1, 4))
    story.append(Paragraph(B("4b. Groups of images") + " — is a whole batch of m images from a different source? "
                           "(average AUROC, 3 models)", body))
    rows = [
        ["Method", "10 images", "20 images", "Comment"],
        ["[2] TU, averaged over the batch", B("94.6 ± 0.7"), "96.8 ± 0.5", "Best with very small batches"],
        ["[5] Cloud topology + intrinsic dimension", "94.2 ± 0.5", "98.9 ± 0.0", "Ties TU at 10 images, ahead from 20"],
        ["[6] Distance between persistence diagrams", "92.1 ± 0.4", B("99.0 ± 0.2"), "Near perfect from 20 images"],
        ["[6] Layer-to-layer distance profile", "90.2 ± 0.9", "94.6 ± 1.6", "Paper's main tool; weaker for OOD"],
        ["Softmax confidence, averaged (baseline)", "85.6 ± 6.7", "89.0 ± 8.3", "Lowest and least stable"],
    ]
    story.append(table(rows, [2.45 * inch, 0.85 * inch, 0.85 * inch, W - 4.15 * inch]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(B("[3] pTU statistical test") + " (the paper's own test on groups of images): with 50 images it "
                           "flags every OOD set in 96–100 % of trials, while flagging normal MNIST only 3–5 % of the time "
                           "(the intended 5 % false-alarm rate). All group methods reach about 100 AUROC with 50 images.", body))

    # ------------------------------------------------------------------ 5. takeaways
    story.append(Paragraph("5. Main takeaways", h2))
    num = ParagraphStyle("num", parent=body, leftIndent=12, bulletIndent=0, spaceAfter=2.5)
    takeaways = [
        B("Topology catches what confidence misses. ") + "The network is 94–100 % confident on SVHN, CIFAR-10 and "
        "uniform noise, so the confidence score fails there (AUROC 60–69). TU, pTU and β<sub>0</sub> still detect them "
        "(78–100), and their results barely change between trained models.",
        B("A method must look at the input. ") + "Neural Persistence [1] describes the trained weights, which are the same "
        "for every input, so it cannot tell normal and unusual inputs apart.",
        B("Simple summaries beat complicated ones. ") + "Counting connected pieces (β<sub>0</sub>) works better than the full "
        "persistent-Laplacian spectrum and TU's distance; the extra eigenvalues and loop (H1) features add noise.",
        B("How layers are combined matters. ") + "The first layer is the most informative in [2], [3] and [4], but those "
        "papers average all layers equally. Rescaling each layer before averaging lifts TU from 79 to 87.",
        B("The best single-image score is mostly a distance. ") + "[5]'s top result measures how far an image is from its "
        "nearest training images; its loop-based part alone is worse than guessing.",
        B("For groups of images, everything works. ") + "All group methods reach about 100 AUROC with 50 images; pTU's "
        "test is the only one with a controlled false-alarm rate.",
    ]
    for i, tk in enumerate(takeaways, 1):
        story.append(Paragraph(tk, num, bulletText=f"{i}."))

    # ------------------------------------------------------------------ 6. limitations
    story.append(Paragraph("6. Limitations and next steps", h2))
    bullets([
        B("Scope: ") + "one small MLP on MNIST (the setting of [1]–[3]; [5] and [6] studied CNNs); 3 trained models.",
        B("My adaptations ") + "(labelled above): single-image versions of [1] and [5], the MLP version of [6], rescaled TU, "
        "and β<sub>0</sub>-only; plus the 0.1 KL scaling.",
        B("Next: ") + "repeat on a CNN (CIFAR-10), test layer-weighted TU/pTU, and use a two-sided pTU score so "
        "low-variance inputs such as Gaussian noise are caught.",
    ])
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "References: [1] Rieck et al., ICLR 2019. [2] Lacombe et al., IJCAI 2021. [3] Yeh &amp; Yang, arXiv:2511.18813. "
        "[4] Wolf et al., arXiv:2509.20220; Jones &amp; Wei (PETLS); Botnan &amp; Dong, arXiv:2408.16741. "
        "[5] Magai &amp; Ayzenberg, arXiv:2204.08624. [6] Purvine et al., AAAI 2023.", small))

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title="Topology of Neural-Network Layers for OOD Detection", author="Dat Bui")
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
