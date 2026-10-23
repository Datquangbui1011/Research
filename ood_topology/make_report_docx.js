// Build the Word versions of both reports: the plain-language report (OOD_Topology_Report_Explained.docx,
// same content as build_plain() in make_report.py) and the technical report (OOD_Topology_Report.docx,
// same content as build()).
// Usage: npm install docx && node make_report_docx.js
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, ImageRun, LevelFormat, Packer, Paragraph, ShadingType,
  Table, TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const ROOT = __dirname;
const OUT = path.join(ROOT, "..", "OOD_Topology_Report_Explained.docx");
const NAVY = "1F3B63", GOOD = "1B7A3A", OK = "A86A00", BAD = "B3261E", BASE = "555555";
const PAGE_W = 12240, MARGIN = 936, CONTENT_W = PAGE_W - 2 * MARGIN; // 0.65" margins

// "**bold** text" -> runs; {color} applies to bold parts if given
function runs(text, opts = {}) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((part) => {
    const bold = part.startsWith("**");
    return new TextRun({
      text: bold ? part.slice(2, -2) : part,
      bold: bold || opts.bold,
      color: bold && opts.boldColor ? opts.boldColor : opts.color,
      size: opts.size,
    });
  });
}
const para = (text, opts = {}) => new Paragraph({ children: runs(text, opts), spacing: { after: 80 }, ...opts.p });
const bullet = (text) => new Paragraph({ children: runs(text), numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 } });
const numbered = (text) => new Paragraph({ children: runs(text), numbering: { reference: "numbers", level: 0 }, spacing: { after: 40 } });
const heading = (text) => new Paragraph({
  children: [new TextRun({ text, bold: true, color: NAVY, size: 24 })],
  spacing: { before: 200, after: 80 },
});

const border = { style: BorderStyle.SINGLE, size: 4, color: "B0B7C3" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(content, width, { header = false, fill } = {}) {
  const children = Array.isArray(content) ? content : [new Paragraph({
    children: header ? [new TextRun({ text: content, bold: true, color: "FFFFFF", size: 18 })] : runs(content, { size: 18 }),
  })];
  return new TableCell({
    children, borders, width: { size: width, type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 90, right: 90 },
    shading: header ? { fill: NAVY, type: ShadingType.CLEAR, color: "auto" }
      : fill ? { fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
  });
}

function table(rows, widths, greyRows = []) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => cell(c, widths[j], {
        header: i === 0,
        fill: greyRows.includes(i) ? "E6E6E6" : (i % 2 === 0 && i > 0 ? "F2F5F9" : undefined),
      })),
    })),
  });
}

function box(paragraphs) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({
      children: [new TableCell({
        children: paragraphs,
        width: { size: CONTENT_W, type: WidthType.DXA },
        borders: Object.fromEntries(["top", "bottom", "left", "right"].map((k) => [k, { style: BorderStyle.SINGLE, size: 8, color: NAVY }])),
        shading: { fill: "EEF3FA", type: ShadingType.CLEAR, color: "auto" },
        margins: { top: 100, bottom: 80, left: 160, right: 160 },
      })],
    })],
  });
}

// verdict cell: coloured bold verdict + plain explanation
function verdict(word, color, rest = "") {
  return [new Paragraph({ children: [new TextRun({ text: word, bold: true, color, size: 18 }), new TextRun({ text: rest ? " " + rest : "", size: 18 })] })];
}

const children = [];
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "Can the “Shape” of a Neural Network Spot Unfamiliar Inputs?", bold: true, size: 30 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun({ text: "A plain-language report on topology-based out-of-distribution detection  ·  Dat Bui  ·  October 2026  ·  github.com/Datquangbui1011/Research", color: "444444", size: 18 })],
}));

children.push(box([
  new Paragraph({ children: [new TextRun({ text: "The big picture", bold: true, color: NAVY, size: 22 })], spacing: { after: 60 } }),
  para("A neural network trained on handwritten digits (MNIST) will still give an answer if you show it a photo of a shoe or random static, and it is often **very confident** while being wrong. Spotting these unfamiliar inputs is called **out-of-distribution (OOD) detection**."),
  para("Six papers suggest the same idea: instead of trusting the network’s confidence, look at the **shape of its internal activity** and check whether it looks normal. They measure shape with **topology**, the math of how things connect (how many separate pieces, how many loops). I tested all six ideas on the same network to see which ones actually work.", { p: { spacing: { after: 0 } } }),
]));

children.push(heading("1. The network I used"));
children.push(bullet("**Why this network:** I needed one network to test all six ideas fairly. Three of the papers used a small network on MNIST, and one (pTU) gave the exact recipe, so I copied it."));
children.push(bullet("**Input:** a 28×28 digit image (784 pixels)."));
children.push(bullet("**Layers:** 784 → 64 → 16 → 10 neurons; the 10 outputs are the digits 0–9."));
children.push(bullet("**Bayesian:** each weight is a range of likely values instead of one number, so you can create many slightly different versions of the same network. Method [3] needs this."));
children.push(bullet("**Accuracy: 97.2 %**, almost the same as the paper’s 97.4 %. To get there I had to weaken one part of the training loss; the paper does not say what it did, so this is flagged."));

children.push(heading("2. The six approaches"));
children.push(para("All six follow the same recipe: **(1)** turn the network’s activity into a shape, **(2)** measure how that shape connects up, **(3)** compare it with what normal training images produce, **(4)** if it looks unusual, flag the input as OOD."));
children.push(table([
  ["Paper", "What it looks at, in plain English"],
  ["[1] Neural Persistence", "Only the network’s weights (its “wiring”), not the input itself."],
  ["[2] Topological Uncertainty (TU)", "Each image lights up the network in a pattern. Compare that pattern’s shape with the usual pattern for that digit."],
  ["[3] pTU", "Run the image through 30 slightly different versions of the network. If their patterns disagree a lot, the network is unsure about this image."],
  ["[4] Persistent Laplacian", "The same patterns as [2], measured with a matrix. One part counts connected pieces; the other adds extra “geometry” detail."],
  ["[5] Data manifold", "Put many images together as a cloud of points and measure the cloud’s clusters, loops and dimension."],
  ["[6] CNN activations", "Also uses point clouds, but compares their shapes using a distance between them."],
], [2800, CONTENT_W - 2800]));

children.push(heading("3. How I tested them"));
children.push(bullet("**Normal images:** MNIST digits."));
children.push(bullet("**Unfamiliar images:** clothes, Japanese characters, photos, house numbers, two kinds of random noise, and digits with some pixels flipped."));
children.push(bullet("**Score (AUROC):** how well a method ranks unfamiliar images above normal ones. **50 = coin flip, 100 = perfect.**"));
children.push(bullet("**Two settings:** one image at a time (“is this image strange?”) and a group of images (“is this whole batch from a different source?”)."));
children.push(bullet("**Comparison point (baseline):** the network’s own confidence, the usual simple method."));
children.push(bullet("**Fairness:** I trained 3 separate copies of the network and averaged the results. The “±” shows how much results change between copies."));

children.push(heading("4. Results"));
children.push(para("**One image at a time** — ranked from best to worst (grey rows = baselines for comparison):", { p: { keepNext: true } }));
const W4 = [400, 3300, 1250, CONTENT_W - 4950];
children.push(table([
  ["#", "Method", "Score", "What it means"],
  ["1", "[5] Point cloud (single-image version)", "**96.5 ± 0.4**", verdict("Best.", GOOD, "The catch: it mostly measures “how far is this image from the training images”.")],
  ["2", "Nearest-neighbour distance (baseline)", "88.4 ± 0.9", verdict("Strong simple baseline.", BASE)],
  ["3", "[2] TU with my fix", "87.4 ± 0.6", verdict("Good.", GOOD, "8 points better than the paper’s version.")],
  ["4", "[4] Counting connected pieces only", "85.8 ± 0.1", verdict("Good", GOOD, "and very stable.")],
  ["5", "[3] pTU", "83.0 ± 5.2", verdict("Good but unstable.", OK, "Changes between network copies; misses random noise.")],
  ["6", "[4] Full version", "82.6 ± 0.8", verdict("OK.", OK, "Adding the extra “geometry” made it worse.")],
  ["7", "[2] TU as in the paper", "79.1 ± 0.8", verdict("OK.", OK, "Beats confidence.")],
  ["8", "Network confidence (baseline)", "74.1 ± 4.8", verdict("Weak.", BASE, "Easily fooled.")],
  ["9", "[1] Neural Persistence", "55.4 ± 5.0", verdict("Fails.", BAD, "Close to a coin flip.")],
], W4, [2, 8]));

const imgW = 624, imgH = Math.round(624 * 750 / 2100);
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
  children: [new ImageRun({
    type: "png", data: fs.readFileSync(path.join(ROOT, "results", "summary_seeds.png")),
    transformation: { width: imgW, height: imgH },
    altText: { title: "Average AUROC per method", description: "Bar chart of average AUROC with error bars over 3 seeds", name: "summary" },
  })],
}));
children.push(para("Figure 1. Average score for every method (mean ± spread over 3 network copies). Grey = baselines, blue = topology methods; the dotted line is a coin flip (50).", { size: 16, color: "444444" }));
children.push(para("**Groups of images:** almost every method gets close to perfect once there are 20–50 images (for example 99 out of 100 with 20 images). pTU’s statistical test is the only one that also keeps false alarms on normal data at the intended level of about 5 %."));

children.push(heading("5. Main takeaways"));
[
  "**Topology catches what confidence misses.** The network was 94–100 % sure about photos and noise it had never seen; the topology methods still flagged them.",
  "**A method must look at the input.** Neural Persistence only looks at the wiring, which is the same for every image, so it cannot tell images apart.",
  "**Simpler works better.** Just counting connected pieces beat the more complicated versions; the extra details added noise instead of useful signal.",
  "**How the layers are combined matters.** The first layer carries the most useful information, but the papers treat all layers equally. Rescaling each layer before averaging raised TU from 79 to 87.",
  "**The best single-image method is mostly a distance check.** It works because unfamiliar images sit far from the training images, not because of deeper shape information.",
  "**With groups of images, almost everything works.**",
].forEach((t) => children.push(numbered(t)));

children.push(heading("6. Limitations and next steps"));
children.push(bullet("I tested only one small network on simple digit images; papers [5] and [6] used bigger image networks (CNNs)."));
children.push(bullet("Some versions are my own adaptations where the original method did not fit (for example, making [1] work per image). They are labelled in the technical report."));
children.push(bullet("**Next:** repeat on a larger network with photos (CIFAR-10), weight the layers in TU and pTU instead of averaging them equally, and fix pTU so it also flags images that are unusually “calm” (which is how it misses random noise)."));
children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
children.push(box([para("**In one sentence:** looking at the shape of a network’s internal activity is a better way to spot unfamiliar inputs than trusting its confidence — but the simplest shape measurements work best, and the top method mainly works by measuring distance to the training data.", { p: { spacing: { after: 0 } } })]));
children.push(new Paragraph({
  spacing: { before: 160 },
  children: [new TextRun({ text: "Papers: [1] Rieck et al., ICLR 2019. [2] Lacombe et al., IJCAI 2021. [3] Yeh & Yang, arXiv:2511.18813. [4] Wolf et al., arXiv:2509.20220; Jones & Wei (PETLS); Botnan & Dong, arXiv:2408.16741. [5] Magai & Ayzenberg, arXiv:2204.08624. [6] Purvine et al., AAAI 2023.", size: 16, color: "444444" })],
}));

function writeDoc(children, out, title) {
  const doc = new Document({
    creator: "Dat Bui",
    title,
    styles: { default: { document: { run: { font: "Arial", size: 19 } } } },
    numbering: {
      config: [
        { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] },
        { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 300 } } } }] },
      ],
    },
    sections: [{
      properties: { page: { size: { width: PAGE_W, height: 15840 }, margin: { top: 720, bottom: 720, left: MARGIN, right: MARGIN } } },
      children,
    }],
  });
  return Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log(out); });
}

function figure() {
  const imgW = 624, imgH = Math.round(624 * 750 / 2100);
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(path.join(ROOT, "results", "summary_seeds.png")),
      transformation: { width: imgW, height: imgH },
      altText: { title: "Average AUROC per method", description: "Bar chart of average AUROC with error bars over 3 seeds", name: "summary" },
    })],
  });
}

const REFS = "[1] Rieck et al., ICLR 2019. [2] Lacombe et al., IJCAI 2021. [3] Yeh & Yang, arXiv:2511.18813. [4] Wolf et al., arXiv:2509.20220; Jones & Wei (PETLS); Botnan & Dong, arXiv:2408.16741. [5] Magai & Ayzenberg, arXiv:2204.08624. [6] Purvine et al., AAAI 2023.";

// ------------------------------------------------------------------ technical report (same content as build() in make_report.py)
function technical() {
  const c = [];
  c.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "Topology of Neural-Network Layers for Out-of-Distribution Detection", bold: true, size: 30 })],
  }));
  c.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: "Dat Bui  ·  October 2026  ·  code and full results: github.com/Datquangbui1011/Research", color: "444444", size: 18 })],
  }));
  c.push(box([
    new Paragraph({ children: [new TextRun({ text: "At a glance", bold: true, color: NAVY, size: 22 })], spacing: { after: 60 } }),
    para("**Task:** find a model used in the six papers of the literature report, implement all six approaches on it, use them to detect out-of-distribution (OOD) inputs, and compare how well each works."),
    para("**Model:** the Bayesian MLP 784→64→16→10 from the pTU paper [3], trained on MNIST (97.2 % test accuracy, close to the paper’s 97.4 %)."),
    para("**Test:** can each method tell MNIST digits apart from 7 kinds of unfamiliar images? Every result is averaged over 3 independently trained models."),
    para("**Main result:** 5 of the 6 topological approaches beat the standard confidence baseline. The best one, [5], reaches 96.5 / 100 AUROC versus 74.1 for confidence. Neural Persistence [1] cannot detect OOD inputs at all.", { p: { spacing: { after: 0 } } }),
  ]));

  c.push(heading("1. The shared model"));
  [
    "**Why this model:** papers [1], [2] and [3] all study small fully connected networks (MLPs) on MNIST, and [3] gives the exact architecture, prior and training rule, so the setup can be reproduced.",
    "**What it is:** 784 inputs (28×28 pixels) → 64 → 16 → 10 classes, ReLU, trained as a Bayesian network (Bayes-by-Backprop) with [3]’s scale-mixture prior (σ₁ = 1, σ₂ = 0.007) and early stopping (patience 25), so each weight has a distribution, not a single value.",
    "**How it is used:** methods [1], [2], [4], [5], [6] use the posterior-mean weights; [3] (pTU) uses 30 posterior weight samples, as in the paper.",
    "**One change from the paper:** the KL term of the loss is scaled by 0.1. With the full term, accuracy stalls at 94.5 %, and [3] does not say how it weighted this term.",
  ].forEach((t) => c.push(bullet(t)));

  c.push(heading("2. The six approaches"));
  c.push(para("All six turn a layer of the network into a shape (a graph or a cloud of points), track how that shape connects up as a threshold changes (persistent homology), and summarize it with numbers. An input whose numbers look unusual compared with training data is flagged as OOD."));
  c.push(table([
    ["Paper", "Object built from the network", "OOD score used here"],
    ["[1] Neural Persistence (Rieck et al. 2019)", "Bipartite layer graph of |W| (weights only); 0-dim superlevel persistence diagram = maximum spanning tree (MST) weights.", "Original NP is input-independent (AUROC = 50 by construction). Adaptation: NP of each input’s activation graph, z-scored per predicted class."],
    ["[2] Topological Uncertainty (Lacombe et al. 2021)", "Activation graph |W(j,i)·x(i)| per layer; persistence diagram = its MST weights.", "Mean over layers of the 2-Wasserstein distance to the class-mean diagram of training data (predicted class)."],
    ["[3] pTU (Yeh & Yang 2025)", "The same activation-graph diagrams under 30 posterior weight samples.", "Fréchet variance of the 30 diagrams, averaged over layers; plus the paper’s permutation test (Alg. 2) on groups of images."],
    ["[4] Persistent Laplacian (Wolf et al. 2025; PETLS)", "In dim 0 with all vertices born first, the persistent Laplacian of Kₛ ⊆ Kₜ equals the graph Laplacian of Kₜ.", "Number of zero eigenvalues (β₀) and nonzero-spectrum features at 3 thresholds, z-scored per predicted class."],
    ["[5] Data manifold (Magai & Ayzenberg 2022)", "Point cloud of layer activations (input, 2 hidden, logits); Vietoris–Rips persistent homology.", "Groups: H0/H1 total persistence + TwoNN intrinsic dimension. Single image (adaptation): change in H0/H1 of its 20 nearest training neighbours when it is added."],
    ["[6] CNN activations (Purvine et al. 2023)", "The same point clouds (an MLP already gives one vector per image); Rips diagrams.", "Groups: sliced-Wasserstein distance to reference diagrams; and the layer-to-layer SW profile."],
  ], [2300, 3500, CONTENT_W - 5800]));

  c.push(heading("3. How the methods were tested"));
  [
    "**Normal data:** 2000 MNIST test digits; reference statistics from MNIST training images.",
    "**OOD data** (2000 each, all 28×28 grayscale in [0, 1]): Fashion-MNIST (clothes), KMNIST (Japanese characters), CIFAR-10 (photos), SVHN (house numbers), Gaussian noise, uniform noise, and MNIST-flip (digits with 100 pixels inverted).",
    "**Score:** AUROC = how well a method ranks OOD images above normal ones; **50 = random guessing, 100 = perfect.** FPR95 (lower is better) = % of OOD images missed when 95 % of normal images are accepted.",
    "**Two settings:** **one image at a time** (the usual OOD task) and **groups of images** (is this whole batch from a different source?), which is how papers [3], [5], [6] use topology.",
    "**Baselines:** softmax confidence (MSP, the baseline in [2]), Bayesian mutual information, and distance to the nearest training images (k-NN).",
    "**Repeats:** every number is mean ± std over 3 independently trained models; † = evaluated on 1000 images per set.",
  ].forEach((t) => c.push(bullet(t)));

  c.push(heading("4. Results"));
  c.push(para("**4a. One image at a time** — average over the 7 OOD sets, mean ± std over 3 trained models, ranked from best to worst. Grey rows are baselines.", { p: { keepNext: true } }));
  c.push(table([
    ["#", "Method", "AUROC ↑", "FPR95 ↓", "Verdict"],
    ["1", "[5] Data manifold, single-image version †", "**96.5 ± 0.4**", "**16.9 ± 2.3**", verdict("Best", GOOD, "— but it mainly measures distance to the training data")],
    ["2", "k-NN distance (baseline) †", "88.4 ± 0.9", "46.4 ± 1.9", verdict("Strong baseline", BASE)],
    ["3", "[2] TU with layers rescaled (my fix)", "87.4 ± 0.6", "60.1 ± 1.6", verdict("Good", GOOD, "— +8 points over the paper’s version")],
    ["4", "[4] Persistent Laplacian, zero eigenvalues only (β₀) †", "85.8 ± 0.1", "50.3 ± 1.6", verdict("Good", GOOD, "— very stable")],
    ["5", "[3] pTU", "83.0 ± 5.2", "39.4 ± 7.0", verdict("Good but unstable", OK, "— misses Gaussian noise")],
    ["6", "[4] Persistent Laplacian, full spectrum †", "82.6 ± 0.8", "59.3 ± 1.5", verdict("OK", OK, "— extra eigenvalues make it worse")],
    ["7", "Bayesian mutual information (baseline)", "81.4 ± 4.8", "60.2 ± 9.0", verdict("Baseline", BASE)],
    ["8", "[2] TU (paper version)", "79.1 ± 0.8", "73.1 ± 3.1", verdict("OK", OK, "— beats confidence")],
    ["9", "Softmax confidence, MSP (baseline)", "74.1 ± 4.8", "74.1 ± 3.0", verdict("Weak baseline", BASE, "— fooled by confident mistakes")],
    ["10", "[1] Neural Persistence (per input)", "55.4 ± 5.0", "89.2 ± 4.9", verdict("Fails", BAD, "— near random guessing")],
  ], [450, 3400, 1100, 1100, CONTENT_W - 6050], [2, 7, 9]));
  c.push(figure());
  c.push(para("Figure 1. Average AUROC (mean ± std over 3 models). Grey = baselines, blue = topological approaches; † = evaluated on 1000 images per set. Dotted line = random guessing.", { size: 16, color: "444444" }));

  c.push(para("**4b. Groups of images** — is a whole batch of m images from a different source? (average AUROC, 3 models)", { p: { keepNext: true, spacing: { before: 120, after: 80 } } }));
  c.push(table([
    ["Method", "10 images", "20 images", "Comment"],
    ["[2] TU, averaged over the batch", "**94.6 ± 0.7**", "96.8 ± 0.5", "Best with very small batches"],
    ["[5] Cloud topology + intrinsic dimension", "94.2 ± 0.5", "98.9 ± 0.0", "Ties TU at 10 images, ahead from 20"],
    ["[6] Distance between persistence diagrams", "92.1 ± 0.4", "**99.0 ± 0.2**", "Near perfect from 20 images"],
    ["[6] Layer-to-layer distance profile", "90.2 ± 0.9", "94.6 ± 1.6", "Paper’s main tool; weaker for OOD"],
    ["Softmax confidence, averaged (baseline)", "85.6 ± 6.7", "89.0 ± 8.3", "Lowest and least stable"],
  ], [3600, 1300, 1300, CONTENT_W - 6200]));
  c.push(para("**[3] pTU statistical test** (the paper’s own test on groups of images): with 50 images it flags every OOD set in 96–100 % of trials, while flagging normal MNIST only 3–5 % of the time (the intended 5 % false-alarm rate). All group methods reach about 100 AUROC with 50 images.", { p: { spacing: { before: 100, after: 80 } } }));

  c.push(heading("5. Main takeaways"));
  [
    "**Topology catches what confidence misses.** The network is 94–100 % confident on SVHN, CIFAR-10 and uniform noise, so the confidence score fails there (AUROC 60–69). TU, pTU and β₀ still detect them (78–100), and their results barely change between trained models.",
    "**A method must look at the input.** Neural Persistence [1] describes the trained weights, which are the same for every input, so it cannot tell normal and unusual inputs apart.",
    "**Simple summaries beat complicated ones.** Counting connected pieces (β₀) works better than the full persistent-Laplacian spectrum and TU’s distance; the extra eigenvalues and loop (H1) features add noise.",
    "**How layers are combined matters.** The first layer is the most informative in [2], [3] and [4], but those papers average all layers equally. Rescaling each layer before averaging lifts TU from 79 to 87.",
    "**The best single-image score is mostly a distance.** [5]’s top result measures how far an image is from its nearest training images; its loop-based part alone is worse than guessing.",
    "**For groups of images, everything works.** All group methods reach about 100 AUROC with 50 images; pTU’s test is the only one with a controlled false-alarm rate.",
  ].forEach((t) => c.push(numbered(t)));

  c.push(heading("6. Limitations and next steps"));
  [
    "**Scope:** one small MLP on MNIST (the setting of [1]–[3]; [5] and [6] studied CNNs); 3 trained models.",
    "**My adaptations** (labelled above): single-image versions of [1] and [5], the MLP version of [6], rescaled TU, and β₀-only; plus the 0.1 KL scaling.",
    "**Next:** repeat on a CNN (CIFAR-10), test layer-weighted TU/pTU, and use a two-sided pTU score so low-variance inputs such as Gaussian noise are caught.",
  ].forEach((t) => c.push(bullet(t)));
  c.push(new Paragraph({ spacing: { before: 160 }, children: [new TextRun({ text: "References: " + REFS, size: 16, color: "444444" })] }));
  return c;
}

writeDoc(children, OUT, "Can the Shape of a Neural Network Spot Unfamiliar Inputs?")
  .then(() => writeDoc(technical(), path.join(ROOT, "..", "OOD_Topology_Report.docx"), "Topology of Neural-Network Layers for OOD Detection"));
