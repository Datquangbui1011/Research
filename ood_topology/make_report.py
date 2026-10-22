"""Build the 2-page PDF report (OOD_Topology_Report.pdf) from the 3-seed results."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "OOD_Topology_Report.pdf"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_DIR / "Arial Bold.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONT_DIR / "Arial Italic.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Arial", fontSize=8.6, leading=10.6, spaceAfter=3)
small = ParagraphStyle("small", parent=body, fontSize=7.4, leading=8.8)
cell = ParagraphStyle("cell", parent=body, fontSize=7.4, leading=8.6, spaceAfter=0)
h1 = ParagraphStyle("h1", parent=body, fontName="Arial-Bold", fontSize=14, leading=17, alignment=TA_CENTER, spaceAfter=2)
sub = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, fontSize=8.6, textColor=colors.HexColor("#444444"), spaceAfter=6)
h2 = ParagraphStyle("h2", parent=body, fontName="Arial-Bold", fontSize=10, leading=12, spaceBefore=5, spaceAfter=2,
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


def build():
    W = letter[0] - 1.3 * inch
    story = []
    story.append(Paragraph("Topology of Neural-Network Layers for Out-of-Distribution Detection", h1))
    story.append(Paragraph("Implementation and evaluation of six topological approaches on one shared model "
                           "&nbsp;·&nbsp; Dat Bui &nbsp;·&nbsp; October 2026 &nbsp;·&nbsp; "
                           "code: github.com/Datquangbui1011/Research", sub))

    story.append(Paragraph("1. Goal and shared model", h2))
    story.append(Paragraph(
        "The goal was to find a model used in the six papers of the literature report, implement every approach on "
        "it, and evaluate each approach as an out-of-distribution (OOD) detector. Papers [1]–[3] all study MLPs on "
        "MNIST, and the pTU paper [3] gives the exact setup. So the shared model is a " + B("Bayesian MLP 784→64→16→10")
        + " (ReLU) trained with Bayes-by-Backprop on MNIST, with [3]'s scale-mixture prior (σ<sub>1</sub> = 1, "
        "σ<sub>2</sub> = 0.007) and early stopping (patience 25). Test accuracy is " + B("97.21 ± 0.09 %")
        + " over 3 seeds ([3] reports 97.39 %). The KL term was scaled by 0.1; with the full KL the model stalls at "
        "94.5 %, and [3] does not state its weighting. Deterministic methods use the posterior-mean weights; pTU "
        "uses 30 posterior samples.", body))

    story.append(Paragraph("2. Approaches as implemented", h2))
    rows = [
        ["Paper", "Object built from the network", "OOD score used here"],
        ["[1] Neural Persistence (Rieck 2019)", "Bipartite layer graph of |W| (weights only); 0-dim superlevel PD = MST",
         "Original NP is input-independent (AUROC = 50 by construction). Adaptation: NP of each input's activation graph, "
         "z-scored per predicted class"],
        ["[2] Topological Uncertainty (Lacombe 2021)", "Activation graph |W(j,i)·x(i)| per layer; PD = MST weights",
         "Mean over layers of the 2-Wasserstein distance to the class-mean PD of training data"],
        ["[3] pTU (Yeh & Yang 2025)", "Activation-graph PDs under 30 posterior weight samples",
         "Fréchet variance of the 30 PDs, averaged over layers; plus the paper's permutation test (Alg. 2)"],
        ["[4] Persistent Laplacian (Wolf 2025; PETLS)", "In dim 0 with all vertices born first, Δ<sub>0</sub><super>K<sub>s</sub>,K<sub>t</sub></super> = graph Laplacian of K<sub>t</sub>",
         "β<sub>0</sub> (zero eigenvalues) and nonzero-spectrum features at 3 thresholds, z-scored per predicted class"],
        ["[5] Data manifold (Magai & Ayzenberg 2022)", "Point cloud of layer activations (input, 2 hidden, logits); Rips PH",
         "Batch: H0/H1 total persistence + TwoNN intrinsic dimension. Per image (adaptation): change in H0/H1 of the 20-NN cloud when x is added"],
        ["[6] CNN activations (Purvine 2023)", "Same point clouds (one vector per image already in an MLP); Rips PDs",
         "Batch: sliced-Wasserstein distance to reference PDs; and the layer-to-layer SW profile"],
    ]
    story.append(table(rows, [1.55 * inch, 2.35 * inch, W - 3.9 * inch]))

    story.append(Paragraph("3. Evaluation protocol", h2))
    story.append(Paragraph(
        "In-distribution: 2000 MNIST test images; reference statistics from MNIST training images. OOD sets (2000 "
        "each, grayscale 28×28 in [0, 1]): Fashion-MNIST, KMNIST, CIFAR-10, SVHN, Gaussian and Uniform noise (as in "
        "[2]), and MNIST-flip (100 pixels inverted, the corruption shift of [2]). Metrics: AUROC (OOD positive) and "
        "FPR95 (% of OOD accepted when 95 % of ID is accepted). Baselines: max-softmax confidence (MSP, the baseline "
        "of [2]), Bayesian mutual information, and k-NN distance in activation space. Every number is the mean ± std "
        "over 3 independently trained models; each seed's metric is first averaged over the 7 OOD sets. "
        "† = first 1000 images per set.", body))

    story.append(Paragraph("4. Results", h2))
    rows = [
        ["Per-image detection", "AUROC ↑", "FPR95 ↓", "Batch level (m = images per batch)", "m = 10", "m = 20"],
        ["MSP (baseline)", "74.1 ± 4.8", "74.1 ± 3.0", "MSP, batch mean", "85.6 ± 6.7", "89.0 ± 8.3"],
        ["Bayesian mutual information", "81.4 ± 4.8", "60.2 ± 9.0", "[2] TU, batch mean", B("94.6 ± 0.7"), "96.8 ± 0.5"],
        ["k-NN distance †", "88.4 ± 0.9", "46.4 ± 1.9", "[5] H0/H1 persistence + intrinsic dim.", "94.2 ± 0.5", "98.9 ± 0.0"],
        ["[1] NP per input (adaptation)", "55.4 ± 5.0", "89.2 ± 4.9", "[6] SW distance to reference PDs", "92.1 ± 0.4", B("99.0 ± 0.2")],
        ["[2] TU (paper)", "79.1 ± 0.8", "73.1 ± 3.1", "[6] layer-to-layer SW profile", "90.2 ± 0.9", "94.6 ± 1.6"],
        ["[2] TU, layers rescaled", "87.4 ± 0.6", "60.1 ± 1.6", "", "", ""],
        ["[3] pTU (paper)", "83.0 ± 5.2", "39.4 ± 7.0", B("[3] pTU permutation test"), B("m = 20"), B("m = 50")],
        ["[4] full persistent Laplacian †", "82.6 ± 0.8", "59.3 ± 1.5", "OOD sets flagged (α = 0.05)", "66–100 %", "96–100 %"],
        ["[4] β<sub>0</sub> only (PH part) †", "85.8 ± 0.1", "50.3 ± 1.6", "MNIST test flagged (false alarm)", "5 ± 2 %", "3 ± 2 %"],
        ["[5] local ΔH0 (adaptation) †", B("96.5 ± 0.4"), B("16.9 ± 2.3"), "", "", ""],
    ]
    story.append(table(rows, [1.75 * inch, 0.72 * inch, 0.72 * inch, 2.05 * inch, 0.72 * inch, W - 5.96 * inch]))
    story.append(Spacer(1, 4))
    story.append(Image(str(ROOT / "results" / "summary_seeds.png"), width=0.86 * W, height=0.86 * W * 750 / 2100))
    story.append(Paragraph("Figure 1. Average AUROC over the 7 OOD sets (mean ± std, 3 seeds). Grey: baselines; "
                           "blue: topological approaches.", small))

    rows = [
        ["Per-image AUROC", "Fashion-MNIST", "KMNIST", "CIFAR-10", "SVHN", "Gaussian", "Uniform", "MNIST-flip"],
        ["MSP", "68.1 ± 4.4", "84.1 ± 0.3", "68.8 ± 6.2", "64.9 ± 11.0", "92.7 ± 0.7", "59.8 ± 13.0", "80.2 ± 1.2"],
        ["[2] TU", "82.2 ± 1.4", "71.2 ± 1.7", "80.6 ± 1.4", "82.2 ± 0.7", "95.3 ± 2.0", "79.5 ± 7.9", "62.6 ± 1.3"],
        ["[3] pTU", "82.1 ± 4.1", "82.7 ± 4.9", "94.5 ± 2.3", "80.7 ± 4.8", "46.9 ± 18.2", "99.9 ± 0.1", "94.5 ± 2.1"],
        ["[4] β<sub>0</sub> †", "84.6 ± 1.3", "78.2 ± 1.7", "77.8 ± 2.0", "84.8 ± 2.3", "98.2 ± 1.7", "96.6 ± 0.5", "80.7 ± 2.1"],
        ["[5] local ΔH0 †", "93.8 ± 0.8", "95.3 ± 0.3", "99.2 ± 0.1", "94.5 ± 1.0", "93.5 ± 1.0", "100.0 ± 0.0", "99.3 ± 0.1"],
    ]
    story.append(Spacer(1, 4))
    story.append(table(rows, [0.95 * inch] + [(W - 0.95 * inch) / 7] * 7))

    story.append(Paragraph("5. Findings", h2))
    findings = [
        B("Topology beats confidence where the network is confidently wrong. ") +
        "SVHN, CIFAR-10 and uniform noise receive softmax confidence of about 0.94–1.0, so MSP drops to 60–69 AUROC, "
        "while TU, pTU and β<sub>0</sub> reach 78–100. Every approach except [1] beats MSP on average, and the "
        "topological scores are also far more stable across training runs (MSP ± 4.8 vs ± 0.1–1.4 for most).",
        B("[1] Neural persistence cannot detect OOD inputs. ") +
        "It summarizes the trained weights, not an input. Applied per input it stays near chance (55.4).",
        B("[2] TU reproduces its paper's claim but is limited by how layers are combined. ") +
        "The first layer alone reaches about 90 AUROC; averaging raw distances over layers lets large-scale layers "
        "dominate. Rescaling each layer by its training mean lifts TU from 79.1 to 87.4 on every seed.",
        B("[3] pTU is the best paper-defined per-image score (83.0) and its permutation test works as intended: ") +
        "OOD sets are flagged in 96–100 % of repeats with 50 images, at the nominal 3–5 % false-alarm rate. "
        "Weakness: pTU can move in either direction. Gaussian noise often gets lower pTU than MNIST "
        "(46.9 ± 18.2), so a one-sided threshold misses it.",
        B("[4] The persistent Laplacian's extra eigenvalues do not help. ") +
        "On these triangle-free layer graphs the 0-dim persistent Laplacian is a graph Laplacian. Its harmonic part "
        "β<sub>0</sub> alone (85.8) beats the full spectrum (82.6) and TU (79.1); the nonzero eigenvalues take a few "
        "discrete values on small unweighted graphs and add noise.",
        B("[5]/[6] Point-cloud topology is the strongest signal. ") +
        "At batch level, H0/H1 persistence plus intrinsic dimension ([5]) and SW distances between diagrams ([6]) "
        "reach 99 AUROC with 20 images; TU's batch mean ties at 10 images. MNIST's intrinsic dimension shrinks layer by "
        "layer (10.6 → 4.6), as [5] reports. Per image, the local ΔH0 score is best overall (96.5 ± 0.4), but it is "
        "essentially a refined nearest-neighbour distance across layers: higher-order (H1) features alone are below "
        "chance, and [6]'s layer-to-layer profile is weaker than the plain distance to reference diagrams.",
    ]
    for f in findings:
        story.append(Paragraph("• " + f, body))
    story.append(Paragraph(B("Limitations. ") +
                           "One small MLP on MNIST (the setting of [1]–[3], not the CNNs of [5]–[6]); 3 seeds; KL "
                           "term scaled by 0.1; per-image versions of [1] and [5], the MLP version of [6], rescaled TU "
                           "and β<sub>0</sub>-only are my adaptations. Next steps: CNN backbones (CIFAR-10), "
                           "layer-weighted TU/pTU, and two-sided pTU scores.", body))
    story.append(Paragraph(
        "[1] Rieck et al., ICLR 2019. [2] Lacombe et al., IJCAI 2021. [3] Yeh &amp; Yang, arXiv:2511.18813. "
        "[4] Wolf et al., arXiv:2509.20220; Jones &amp; Wei (PETLS); Botnan &amp; Dong, arXiv:2408.16741. "
        "[5] Magai &amp; Ayzenberg, arXiv:2204.08624. [6] Purvine et al., AAAI 2023.", small))

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                            topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                            title="Topology of Neural-Network Layers for OOD Detection", author="Dat Bui")
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
