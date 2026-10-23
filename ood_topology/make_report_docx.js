// Build the plain-language report as a Word document (OOD_Topology_Report_Explained.docx).
// Same content as build_plain() in make_report.py.
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

const doc = new Document({
  creator: "Dat Bui",
  title: "Can the Shape of a Neural Network Spot Unfamiliar Inputs?",
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

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log(OUT); });
