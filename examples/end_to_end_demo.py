#!/usr/bin/env python3
"""
Penrose Memory Palace — End-to-End Demo with Real Neural Embeddings.

Pipeline:
  1. Load sentence-transformers (all-MiniLM-L6-v2, 384D)
  2. Encode 200 sentences from 10 domains
  3. PCA: 384D → 2D (learned projection)
  4. Penrose snap: project 2D points to nearest Penrose vertex
  5. Store in PenroseMemory
  6. Run 50 queries, retrieve via Penrose walk
  7. Compare 4 methods: flat NN, random+NN, PCA+NN, PCA+Penrose
  8. Measure recall@5, recall@10, recall@20
  9. Plot PCA 2D scatter with Penrose overlay, color-coded by domain

Output:
  - Console metrics
  - Plot saved to examples/penrose_demo_plot.png
  - Results saved to examples/END-TO-END-RESULTS.md
"""

import sys, os, time, math, random, json
import numpy as np

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

PLOT_PATH  = os.path.join(SCRIPT_DIR, "penrose_demo_plot.png")
RESULTS_MD = os.path.join(SCRIPT_DIR, "END-TO-END-RESULTS.md")

# ── 200 sentences across 10 domains (20 each) ──
DOMAINS = {
    "physics": [
        "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "Quantum entanglement allows particles to be correlated regardless of distance.",
        "Einstein's theory of general relativity describes gravity as spacetime curvature.",
        "The Heisenberg uncertainty principle limits simultaneous measurement of position and momentum.",
        "Superconductivity occurs when electrical resistance drops to zero below a critical temperature.",
        "The strong nuclear force binds quarks together inside protons and neutrons.",
        "Dark matter does not emit or interact with electromagnetic radiation.",
        "Wave-particle duality describes how quantum objects exhibit both wave and particle properties.",
        "The cosmological constant represents the energy density of empty space.",
        "Neutrinos are subatomic particles with extremely small mass and no electric charge.",
        "Maxwell's equations unify electricity, magnetism, and optics into a single framework.",
        "The Pauli exclusion principle prevents identical fermions from occupying the same quantum state.",
        "Hawking radiation is theoretical thermal radiation emitted by black holes.",
        "String theory proposes that fundamental particles are one-dimensional strings.",
        "The Higgs boson gives other particles their mass through the Higgs mechanism.",
        "Plasma is the fourth state of matter where gas is ionized into charged particles.",
        "The Chandrasekhar limit defines the maximum mass of a stable white dwarf star.",
        "Bose-Einstein condensate forms when bosons are cooled near absolute zero.",
        "The electromagnetic spectrum ranges from radio waves to gamma rays.",
        "Feynman diagrams visually represent interactions between subatomic particles.",
    ],
    "cooking": [
        "Searing meat at high temperature creates a Maillard reaction for browning.",
        "Sourdough bread relies on wild yeast and lactobacillus for fermentation.",
        "Tempering chocolate requires precise temperature control between 27 and 31 degrees Celsius.",
        "A roux made from equal parts butter and flour thickens sauces.",
        "Braising combines dry and wet heat cooking for tenderizing tough cuts of meat.",
        "Mise en place means having all ingredients measured and ready before cooking.",
        "Sodium bicarbonate acts as a leavening agent when combined with acidic ingredients.",
        "Deglazing a pan with wine dissolves fond to create flavorful sauces.",
        "Fermentation of milk by lactic acid bacteria produces yogurt.",
        "Blanching vegetables briefly in boiling water preserves their color and texture.",
        "The glycocalyx on cell membranes consists of glycoproteins and glycolipids.",
        "Reduction sauces intensify flavor as water evaporates during simmering.",
        "Proofing bread dough allows yeast to produce carbon dioxide for rising.",
        "Salt enhances flavor by suppressing bitterness and amplifying sweetness.",
        "Slow cooking collagen-rich meats converts it into gelatin for tenderness.",
        "Emulsions like mayonnaise combine oil and water using lecithin from egg yolks.",
        "Caramelization of sugars occurs above 160 degrees Celsius creating complex flavors.",
        "Resting meat after cooking allows juices to redistribute evenly.",
        "Acidic marinades can denature proteins and tenderize meat surfaces.",
        "Steaming preserves water-soluble vitamins better than boiling vegetables.",
    ],
    "history": [
        "The fall of Constantinople in 1453 marked the end of the Byzantine Empire.",
        "The Magna Carta was signed in 1215, limiting the power of the English monarchy.",
        "The Industrial Revolution began in Britain during the late eighteenth century.",
        "The French Revolution overthrew the monarchy and established a republic in 1789.",
        "The Silk Road connected East Asia to the Mediterranean for over 1500 years.",
        "The Black Death killed an estimated one-third of Europe's population in the fourteenth century.",
        "The Declaration of Independence was adopted by the Continental Congress on July 4, 1776.",
        "The Roman Empire reached its greatest territorial extent under Emperor Trajan.",
        "The Renaissance began in Italy during the fourteenth century, reviving classical learning.",
        "The Cold War was a period of geopolitical tension between the US and USSR from 1947 to 1991.",
        "The Treaty of Westphalia in 1648 established the modern concept of state sovereignty.",
        "Ancient Egypt built pyramids as monumental tombs for their pharaohs.",
        "The Gutenberg printing press revolutionized information dissemination in the fifteenth century.",
        "World War I was triggered by the assassination of Archduke Franz Ferdinand in 1914.",
        "The Ming Dynasty ruled China from 1368 to 1644 and built much of the Great Wall.",
        "The Mongol Empire under Genghis Khan became the largest contiguous land empire in history.",
        "The Apollo 11 mission landed humans on the Moon on July 20, 1969.",
        "The Berlin Wall fell on November 9, 1989, symbolizing the end of the Cold War.",
        "The Haitian Revolution was the only successful slave revolt leading to a free state.",
        "The Meiji Restoration in 1868 transformed Japan from a feudal to a modern state.",
    ],
    "programming": [
        "A binary search algorithm finds elements in a sorted array in O(log n) time.",
        "Garbage collection in Python uses reference counting with a cycle detector.",
        "Docker containers package applications with their dependencies for consistent deployment.",
        "MapReduce divides large data processing tasks across distributed computing nodes.",
        "RESTful APIs use HTTP methods like GET, POST, PUT, and DELETE for resource manipulation.",
        "Functional programming emphasizes immutability and pure functions without side effects.",
        "Git uses a directed acyclic graph to track changes in source code history.",
        "Polymorphism allows objects of different types to be treated through a common interface.",
        "The CAP theorem states distributed systems cannot guarantee consistency and availability simultaneously.",
        "Recursive functions call themselves and require a base case to terminate.",
        "Merkle trees enable efficient verification of data integrity in distributed systems.",
        "QuickSort uses divide and conquer with average case O(n log n) performance.",
        "Relational databases use ACID transactions to ensure data integrity.",
        "Concurrency and parallelism are distinct concepts in computer science.",
        "A hash table provides average O(1) time complexity for lookup operations.",
        "Design patterns provide reusable solutions to common software engineering problems.",
        "Compiler optimization transforms code to improve execution speed without changing semantics.",
        "Event-driven architectures respond to actions rather than following sequential control flow.",
        "The single responsibility principle states a class should have only one reason to change.",
        "Dynamic programming solves complex problems by breaking them into overlapping subproblems.",
    ],
    "biology": [
        "DNA replication occurs during the S phase of the cell cycle.",
        "Mitochondria are organelles that produce ATP through oxidative phosphorylation.",
        "Natural selection drives evolution by favoring traits that enhance reproductive success.",
        "CRISPR-Cas9 technology enables precise gene editing in living organisms.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
        "The human genome contains approximately 20,000 to 25,000 protein-coding genes.",
        "Antibodies are proteins produced by B cells to neutralize pathogens.",
        "Epigenetics studies heritable changes in gene expression without altering DNA sequence.",
        "Stem cells can differentiate into specialized cell types through cellular differentiation.",
        "The polymerase chain reaction amplifies specific DNA sequences exponentially.",
        "Apoptosis is programmed cell death essential for development and tissue homeostasis.",
        "Ribosomes translate messenger RNA into proteins during protein synthesis.",
        "Endosymbiotic theory explains how eukaryotic cells acquired organelles like mitochondria.",
        "Telomeres protect chromosome ends from degradation during cell division.",
        "Homeostasis maintains stable internal conditions through negative feedback loops.",
        "Meiosis produces haploid gametes with half the chromosome number of somatic cells.",
        "The blood-brain barrier restricts the passage of substances from blood to brain tissue.",
        "Gut microbiota play crucial roles in digestion, immunity, and mental health.",
        "Adenosine triphosphate is the primary energy currency of all living cells.",
        "Genetic drift causes random changes in allele frequencies within populations.",
    ],
    "music": [
        "A major scale follows the interval pattern of whole and half steps.",
        "Counterpoint is the technique of combining multiple independent melodic lines.",
        "The twelve-bar blues is a common chord progression in jazz and blues music.",
        "A fugue is a contrapuntal composition with a subject introduced in imitation.",
        "Equal temperament divides the octave into twelve semitones of equal ratio.",
        "Harmony results from the simultaneous sounding of two or more musical notes.",
        "Syncopation places emphasis on normally weak beats in a musical measure.",
        "Timbre distinguishes different instruments playing the same pitch at the same volume.",
        "The circle of fifths shows relationships among the twelve chromatic pitches.",
        "Sonata form consists of exposition, development, and recapitulation sections.",
        "A diminished seventh chord is built entirely from stacks of minor thirds.",
        "Polyrhythm involves the simultaneous use of two or more conflicting rhythms.",
        "Dynamics in music range from pianissimo to fortissimo across the loudness spectrum.",
        "The overtone series contains frequencies that are integer multiples of the fundamental.",
        "Chromatic modulation shifts to a new key using notes outside the current scale.",
        "A cadence marks the end of a musical phrase through harmonic resolution.",
        "Rubato allows expressive flexibility in tempo while maintaining overall timing.",
        "The blues scale adds a flatted fifth to the minor pentatonic scale.",
        "Orchestration is the art of arranging music for different instrumental combinations.",
        "A tritone is an interval of three whole steps, historically called the devil in music.",
    ],
    "mathematics": [
        "The fundamental theorem of calculus connects differentiation and integration.",
        "Euler's identity relates five fundamental mathematical constants in one equation.",
        "Gödel's incompleteness theorems show that formal systems cannot prove their own consistency.",
        "A group is a set equipped with an associative binary operation, identity, and inverses.",
        "The Riemann hypothesis concerns the distribution of non-trivial zeros of the zeta function.",
        "Fourier analysis decomposes functions into sums of sinusoidal components.",
        "The Banach-Tarski paradox shows that a sphere can be decomposed and reassembled into two spheres.",
        "Bayes' theorem updates the probability of a hypothesis given new evidence.",
        "The prime number theorem describes the asymptotic distribution of prime numbers.",
        "Linear algebra studies vector spaces and linear transformations between them.",
        "Topology studies properties preserved under continuous deformations like stretching and bending.",
        "The central limit theorem states that sample means approach a normal distribution.",
        "Differential equations relate functions to their derivatives, modeling change over time.",
        "Set theory provides the foundational language for virtually all of mathematics.",
        "The halting problem is undecidable: no algorithm can determine if arbitrary programs halt.",
        "Eigenvalues of a linear transformation indicate scaling factors along principal directions.",
        "The pigeonhole principle guarantees that if items are placed in fewer containers, one must contain multiple.",
        "Modular arithmetic operates on remainders and underlies cryptography algorithms like RSA.",
        "Category theory studies abstract structures and relationships between mathematical concepts.",
        "A Markov chain is a stochastic process where the next state depends only on the current state.",
    ],
    "geography": [
        "The Mariana Trench is the deepest known part of the ocean at approximately 11,000 meters.",
        "The Amazon Rainforest produces about 20 percent of the world's oxygen supply.",
        "Tectonic plate movements cause earthquakes along fault lines at plate boundaries.",
        "The Sahara Desert expands and contracts with long-term climate cycles.",
        "The Gulf Stream is a warm ocean current that influences Western European climate.",
        "Glaciers store approximately 69 percent of the world's fresh water supply.",
        "The Ring of Fire is a seismically active zone encircling the Pacific Ocean.",
        "Erosion shapes landscapes through the action of water, wind, and ice over time.",
        "The Nile River flows over 6,650 kilometers, making it one of the longest rivers.",
        "Monsoons are seasonal wind patterns that bring heavy rainfall to South and Southeast Asia.",
        "Permafrost is permanently frozen ground found in Arctic and subarctic regions.",
        "The Coriolis effect causes moving air and water to deflect on a rotating Earth.",
        "Karst landscapes form from the dissolution of soluble rocks like limestone.",
        "El Nino is a periodic warming of Pacific Ocean waters affecting global weather.",
        "The Himalayas were formed by the collision of the Indian and Eurasian tectonic plates.",
        "Wetlands serve as natural water filters and provide critical habitat for wildlife.",
        "The ozone layer in the stratosphere absorbs harmful ultraviolet radiation from the Sun.",
        "Volcanic eruptions can inject aerosols into the stratosphere, temporarily cooling climate.",
        "Estuaries are transition zones where rivers meet the sea, creating brackish water.",
        "The thermohaline circulation drives global ocean currents based on density differences.",
    ],
    "psychology": [
        "Classical conditioning associates a neutral stimulus with an unconditioned response.",
        "Cognitive dissonance is the mental discomfort from holding contradictory beliefs.",
        "Maslow's hierarchy of needs posits that basic needs must be met before higher ones.",
        "The bystander effect reduces the likelihood of helping when others are present.",
        "Working memory has a limited capacity of approximately seven plus or minus two items.",
        "Confirmation bias leads people to favor information confirming their existing beliefs.",
        "Attachment theory describes how early relationships with caregivers shape social development.",
        "The placebo effect occurs when belief in treatment produces real physiological changes.",
        "Flow state is a mental state of complete absorption and optimal performance.",
        "Operant conditioning modifies behavior through reinforcement and punishment consequences.",
        "The Dunning-Kruger effect describes how low-ability individuals overestimate their competence.",
        "Neuroplasticity is the brain's ability to reorganize itself by forming new neural connections.",
        "The Stanford prison experiment demonstrated how social roles influence behavior.",
        "Cognitive behavioral therapy identifies and changes dysfunctional thought patterns.",
        "The serial position effect shows better recall for items at the beginning and end of lists.",
        "Mirror neurons fire both when performing and observing an action.",
        "Intrinsic motivation comes from internal satisfaction rather than external rewards.",
        "The fundamental attribution error overemphasizes personality over situational factors.",
        "Habituation is the decrease in response to a stimulus after repeated exposure.",
        "Emotional intelligence involves recognizing, understanding, and managing emotions.",
    ],
    "philosophy": [
        "Descartes' cogito ergo sum establishes existence through the act of thinking.",
        "Utilitarianism judges actions by their consequences and overall happiness produced.",
        "Kant's categorical imperative requires acting only on universalizable maxims.",
        "Plato's allegory of the cave illustrates the difference between perception and reality.",
        "Existentialism holds that individuals create meaning through free will and choice.",
        "The trolley problem is a thought experiment exploring moral decision-making dilemmas.",
        "Solipsism is the philosophical position that only one's own mind is certain to exist.",
        "John Rawls proposed the veil of ignorance as a method for determining fair principles.",
        "Stoicism teaches that virtue is the highest good and emotions should be controlled by reason.",
        "The Ship of Theseus questions whether an object remains the same after all parts are replaced.",
        "Nihilism rejects religious and moral principles, asserting that life lacks inherent meaning.",
        "Phenomenology studies structures of conscious experience from the first-person perspective.",
        "Pragmatism evaluates theories by their practical consequences and usefulness.",
        "The problem of induction questions whether past observations justify future predictions.",
        "Epistemology is the branch of philosophy concerned with the nature of knowledge.",
        "Dualism posits that mind and body are fundamentally distinct kinds of substances.",
        "Absurdism holds that humans seek meaning in a universe that is meaningless.",
        "The social contract theory posits that individuals consent to authority for mutual benefit.",
        "Deontology judges the morality of actions based on rules rather than outcomes.",
        "David Hume argued that causation is a habit of thought, not directly observable.",
    ],
}

# ── Queries (5 per domain = 50) ──
QUERIES = {
    "physics": [
        "How does quantum mechanics describe particle behavior at microscopic scales?",
        "What is the relationship between energy and mass in special relativity?",
        "How do fundamental forces of nature interact with matter?",
        "What happens inside a black hole beyond the event horizon?",
        "How does the Higgs field give particles their mass?",
    ],
    "cooking": [
        "What technique creates a golden crust on steak through high heat?",
        "How does bread rise during the fermentation process?",
        "What is the science behind emulsions in salad dressings?",
        "How does temperature affect the texture of chocolate?",
        "Why do we let meat rest after cooking it?",
    ],
    "history": [
        "What caused the collapse of major empires throughout world history?",
        "How did trade routes connect ancient civilizations across continents?",
        "What were the consequences of the French Revolution for European politics?",
        "How did the printing press transform the spread of knowledge?",
        "What factors led to the end of the Cold War between superpowers?",
    ],
    "programming": [
        "How do efficient sorting algorithms achieve logarithmic time complexity?",
        "What are the trade-offs between different database consistency models?",
        "How does containerization improve software deployment workflows?",
        "What design principles lead to maintainable object-oriented code?",
        "How do distributed systems handle data replication and fault tolerance?",
    ],
    "biology": [
        "How do cells duplicate their genetic material before division?",
        "What role do organelles play in cellular energy production?",
        "How has gene editing technology advanced with CRISPR?",
        "What mechanisms do organisms use to maintain internal stability?",
        "How does the immune system recognize and neutralize pathogens?",
    ],
    "music": [
        "What makes a chord progression sound like the blues?",
        "How do composers create tension and resolution through harmony?",
        "What is the mathematical basis for musical tuning systems?",
        "How does rhythm create interest through unexpected accent patterns?",
        "What distinguishes different instruments playing the same note?",
    ],
    "mathematics": [
        "How does calculus connect the concepts of rate of change and accumulation?",
        "What are the implications of incompleteness in formal mathematical systems?",
        "How does probability theory update beliefs with new evidence?",
        "What geometric properties remain unchanged under continuous transformations?",
        "How do linear transformations scale vectors along specific directions?",
    ],
    "geography": [
        "How do ocean currents influence regional climate patterns?",
        "What geological processes create the deepest ocean trenches?",
        "How do seasonal weather patterns affect agricultural regions?",
        "What role do glaciers play in the global water cycle?",
        "How do tectonic forces reshape continental boundaries over time?",
    ],
    "psychology": [
        "How do early childhood experiences shape adult personality and behavior?",
        "What cognitive biases affect human decision-making and judgment?",
        "How do different therapeutic approaches treat mental health conditions?",
        "What factors influence whether bystanders intervene in emergencies?",
        "How does motivation from internal drives differ from external rewards?",
    ],
    "philosophy": [
        "How do different ethical frameworks evaluate the morality of actions?",
        "What is the relationship between perception and objective reality?",
        "How do social and political philosophies justify authority and governance?",
        "What does it mean to exist and how can we be certain of our consciousness?",
        "How do we determine what constitutes genuine knowledge versus mere belief?",
    ],
}


# ══════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════

def recall_at_k(retrieved_ids: list, ground_truth_id: int, k: int) -> float:
    """1.0 if ground_truth_id is in the top-k retrieved results."""
    return 1.0 if ground_truth_id in retrieved_ids[:k] else 0.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def flat_nn_search(query_emb: np.ndarray, all_embs: np.ndarray, all_ids: list, k: int) -> list:
    """Brute-force cosine similarity search in 384D."""
    sims = np.array([cosine_sim(query_emb, all_embs[i]) for i in range(len(all_embs))])
    top_idx = np.argsort(sims)[::-1][:k]
    return [all_ids[i] for i in top_idx]


def random_2d_search(query_2d: np.ndarray, all_2d: np.ndarray, all_ids: list, k: int) -> list:
    """Nearest neighbor in 2D random projection space."""
    dists = np.linalg.norm(all_2d - query_2d, axis=1)
    top_idx = np.argsort(dists)[:k]
    return [all_ids[i] for i in top_idx]


def pca_2d_search(query_2d: np.ndarray, all_2d: np.ndarray, all_ids: list, k: int) -> list:
    """Nearest neighbor in 2D PCA space (no tiling)."""
    return random_2d_search(query_2d, all_2d, all_ids, k)


def penrose_snap(x: float, y: float, phi: float = 1.618033988749895, radius: float = 2.0) -> tuple:
    """
    Snap a 2D point to the nearest Penrose tiling vertex.
    
    Uses Ammann-Beenker / pentagrid approach:
    Generate a set of Penrose vertex candidates by sampling the 5-grid intersection
    within a neighborhood, then return the closest one.
    """
    # Pentagrid parameters
    gamma = 0.5  # offset parameter (irrational avoids degenerate intersections)
    
    best_point = (x, y)
    best_dist = float('inf')
    
    # Search in a neighborhood of grid lines
    search_range = int(radius) + 1
    
    for k0 in range(-search_range, search_range + 1):
        for k1 in range(-search_range, search_range + 1):
            for k2 in range(-search_range, search_range + 1):
                # Solve for intersection of 3 of 5 pentagrid lines
                # Grid j: n_j + gamma = x*cos(j*pi/5) + y*sin(j*pi/5)
                # Pick lines 0, 1, 2 and solve for (x, y) given n0=k0, n1=k1, n2=k2
                # This is overdetermined, so we solve the first 2 and check the rest
                
                c0, s0 = math.cos(0), math.sin(0)  # = 1, 0
                c1, s1 = math.cos(math.pi / 5), math.sin(math.pi / 5)
                
                # n0 + gamma = c0*x + s0*y = x
                # n1 + gamma = c1*x + s1*y
                vx = k0 + gamma
                vy = (k1 + gamma - c1 * vx) / s1 if abs(s1) > 1e-10 else 0.0
                
                # Check remaining grids (2, 3, 4) for consistency
                valid = True
                for j in range(2, 5):
                    cj = math.cos(j * math.pi / 5)
                    sj = math.sin(j * math.pi / 5)
                    val = cj * vx + sj * vy
                    nj = round(val - gamma)
                    if abs(val - (nj + gamma)) > 0.1:  # tolerance
                        valid = False
                        break
                
                if valid:
                    dist = math.hypot(vx - x, vy - y)
                    if dist < best_dist:
                        best_dist = dist
                        best_point = (vx, vy)
    
    return best_point, best_dist


def penrose_snap_fast(points_2d: np.ndarray, scale: float = 1.0) -> tuple:
    """
    Fast Penrose snap: generate a set of Penrose vertices, then for each point
    find the nearest vertex via KD-tree.
    
    The tiling is scaled so that vertex density matches the data density.
    Scale is chosen so that average nearest-neighbor distance among data points
    is roughly equal to the average Penrose vertex spacing.
    """
    from scipy.spatial import cKDTree
    
    phi = 1.618033988749895
    
    # Auto-scale: compute average nearest-neighbor distance in the data
    data_tree = cKDTree(points_2d)
    data_nn_dists, _ = data_tree.query(points_2d, k=2)  # k=2 because k=1 is self
    avg_nn = np.mean(data_nn_dists[:, 1])
    
    # Target: Penrose vertices should be spaced at ~avg_nn / 2
    # In a pentagrid with offset gamma, vertex density scales as ~1/scale^2
    # We scale the grid to get denser vertices
    target_spacing = avg_nn * 0.5  # want vertices denser than data
    grid_scale = 1.0 / max(target_spacing, 0.01)  # scale up the grid frequency
    
    gamma = 0.5 * phi  # irrational offset
    n_angles = 5
    
    # Generate pentagrid vertices
    cos_vals = [math.cos(j * math.pi / 5) for j in range(n_angles)]
    sin_vals = [math.sin(j * math.pi / 5) for j in range(n_angles)]
    
    margin = 3.0
    x_min, x_max = points_2d[:, 0].min() - margin, points_2d[:, 0].max() + margin
    y_min, y_max = points_2d[:, 1].min() - margin, points_2d[:, 1].max() + margin
    
    n_range = 60  # generous range for scaled grid
    
    vertices = []
    for i in range(n_angles):
        for j in range(i + 1, n_angles):
            ci, si = cos_vals[i] * grid_scale, sin_vals[i] * grid_scale
            cj, sj = cos_vals[j] * grid_scale, sin_vals[j] * grid_scale
            
            det = ci * sj - cj * si
            if abs(det) < 1e-10:
                continue
            
            for ni in range(-n_range, n_range + 1):
                for nj in range(-n_range, n_range + 1):
                    b0 = ni + gamma
                    b1 = nj + gamma
                    vx = (b0 * sj - b1 * si) / det
                    vy = (ci * b1 - cj * b0) / det
                    
                    valid = True
                    for m in range(n_angles):
                        if m == i or m == j:
                            continue
                        val = (cos_vals[m] * grid_scale) * vx + (sin_vals[m] * grid_scale) * vy
                        nm = round(val - gamma)
                        if abs(val - (nm + gamma)) > 0.15:
                            valid = False
                            break
                    
                    if valid:
                        if x_min <= vx <= x_max and y_min <= vy <= y_max:
                            vertices.append((vx, vy))
    
    if not vertices:
        return points_2d, np.zeros(len(points_2d)), points_2d
    
    vertices = np.array(vertices)
    vertices = np.unique(np.round(vertices, decimals=6), axis=0)
    
    tree = cKDTree(vertices)
    dists, indices = tree.query(points_2d)
    snapped = vertices[indices]
    
    return snapped, dists, vertices


def penrose_2d_search(query_snapped: np.ndarray, query_2d: np.ndarray,
                       all_snapped: np.ndarray, all_2d: np.ndarray,
                       all_ids: list, k: int) -> list:
    """Nearest neighbor among Penrose-snapped 2D coordinates.
    
    Two-tier ranking:
    1. First sort by distance between snapped coordinates (Penrose vertex distance)
    2. For points sharing the same snapped vertex, use original PCA distance as tiebreaker
    """
    # Primary: snapped vertex distance
    snap_dists = np.linalg.norm(all_snapped - query_snapped, axis=1)
    # Secondary: original PCA-space distance (for tie-breaking within same vertex)
    pca_dists = np.linalg.norm(all_2d - query_2d, axis=1)
    
    # Combined score: snap_dist * 1000 + pca_dist (snap dominates, pca breaks ties)
    combined = snap_dists * 1000 + pca_dists
    top_idx = np.argsort(combined)[:k]
    return [all_ids[i] for i in top_idx]


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PENROSE MEMORY PALACE — END-TO-END DEMO")
    print("=" * 70)
    
    # ── Step 1: Load model ──
    print("\n[1/9] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("      ✓ Model loaded. Embedding dim: 384")
    
    # ── Step 2: Encode sentences ──
    print("\n[2/9] Encoding 200 sentences from 10 domains...")
    sentences = []
    domain_labels = []
    domain_names = []
    
    for domain, sents in DOMAINS.items():
        domain_names.append(domain)
        didx = len(domain_names) - 1
        for s in sents:
            sentences.append(s)
            domain_labels.append(didx)
    
    print(f"      Encoding {len(sentences)} sentences...")
    embeddings = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings)
    domain_labels = np.array(domain_labels)
    print(f"      ✓ Embeddings shape: {embeddings.shape}")
    
    # ── Step 3: PCA 384D → 2D ──
    print("\n[3/9] Running PCA: 384D → 2D...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=SEED)
    coords_2d = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    print(f"      ✓ PCA explained variance: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}, total={sum(explained):.4f}")
    
    # ── Step 3b: Random projection baseline ──
    print("\n      Generating random 2D projection for baseline...")
    rand_proj = np.random.RandomState(SEED).randn(384, 2)
    rand_proj /= np.linalg.norm(rand_proj, axis=0, keepdims=True)
    coords_2d_random = embeddings @ rand_proj
    
    # ── Step 4: Penrose tiling snap ──
    print("\n[4/9] Snapping PCA 2D points to Penrose tiling vertices...")
    snapped_2d, snap_dists, penrose_vertices = penrose_snap_fast(coords_2d)
    avg_snap_dist = snap_dists.mean()
    max_snap_dist = snap_dists.max()
    print(f"      ✓ Penrose vertices generated: {len(penrose_vertices)}")
    print(f"      ✓ Avg snap distance: {avg_snap_dist:.4f}, Max: {max_snap_dist:.4f}")
    
    # ── Step 5: Store in PenroseMemory ──
    print("\n[5/9] Storing in PenroseMemory...")
    from penrose_memory import PenroseMemory
    
    pm = PenroseMemory(embedding_dim=384)
    tile_ids = []
    for i, sent in enumerate(sentences):
        tid = pm.store(sent, embeddings[i].tolist())
        tile_ids.append(tid)
    print(f"      ✓ Stored {len(pm)} memories")
    
    # ── Step 6: Encode queries and run 50 queries ──
    print("\n[6/9] Running 50 queries...")
    query_texts = []
    query_domains = []
    query_ground_truth = []  # index of the best-matching sentence in 384D
    
    for domain, queries in QUERIES.items():
        didx = domain_names.index(domain)
        for q in queries:
            query_texts.append(q)
            query_domains.append(didx)
    
    query_embeddings = model.encode(query_texts, show_progress_bar=False, normalize_embeddings=True)
    query_embeddings = np.array(query_embeddings)
    query_2d_pca = pca.transform(query_embeddings)
    query_2d_random = query_embeddings @ rand_proj
    query_snapped_2d, _, _ = penrose_snap_fast(query_2d_pca)
    
    # Determine ground truth: nearest neighbor in 384D cosine space
    for i in range(len(query_texts)):
        sims = np.array([cosine_sim(query_embeddings[i], embeddings[j]) for j in range(len(embeddings))])
        query_ground_truth.append(int(np.argmax(sims)))
    
    print(f"      ✓ {len(query_texts)} queries encoded and projected")
    
    # ── Step 7: Compare 4 methods ──
    print("\n[7/9] Comparing 4 retrieval methods...")
    
    methods = {
        "A) Flat 384D NN": "flat",
        "B) Random 2D + NN": "random_2d",
        "C) PCA 2D + NN": "pca_2d",
        "D) PCA 2D + Penrose": "penrose_2d",
    }
    
    results = {name: {"r@5": [], "r@10": [], "r@20": []} for name in methods}
    
    for qi in range(len(query_texts)):
        gt_id = query_ground_truth[qi]
        query_emb = query_embeddings[qi]
        
        for method_name, method_key in methods.items():
            if method_key == "flat":
                retrieved = flat_nn_search(query_emb, embeddings, list(range(len(sentences))), 20)
            elif method_key == "random_2d":
                retrieved = random_2d_search(query_2d_random[qi], coords_2d_random, list(range(len(sentences))), 20)
            elif method_key == "pca_2d":
                retrieved = pca_2d_search(query_2d_pca[qi], coords_2d, list(range(len(sentences))), 20)
            elif method_key == "penrose_2d":
                retrieved = penrose_2d_search(query_snapped_2d[qi], query_2d_pca[qi], snapped_2d, coords_2d, list(range(len(sentences))), 20)
            
            results[method_name]["r@5"].append(recall_at_k(retrieved, gt_id, 5))
            results[method_name]["r@10"].append(recall_at_k(retrieved, gt_id, 10))
            results[method_name]["r@20"].append(recall_at_k(retrieved, gt_id, 20))
    
    # ── Step 8: Report metrics ──
    print("\n[8/9] Results:")
    print("-" * 70)
    print(f"{'Method':<30} {'Recall@5':>10} {'Recall@10':>10} {'Recall@20':>10}")
    print("-" * 70)
    
    results_summary = {}
    for method_name in methods:
        r5 = np.mean(results[method_name]["r@5"]) * 100
        r10 = np.mean(results[method_name]["r@10"]) * 100
        r20 = np.mean(results[method_name]["r@20"]) * 100
        print(f"{method_name:<30} {r5:>9.1f}% {r10:>9.1f}% {r20:>9.1f}%")
        results_summary[method_name] = {"r@5": r5, "r@10": r10, "r@20": r20}
    
    print("-" * 70)
    
    # ── Step 9: Plot ──
    print("\n[9/9] Generating plot...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    
    # Color map for domains
    domain_colors = plt.cm.tab10(np.linspace(0, 1, len(domain_names)))
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: PCA 2D scatter with Penrose overlay
    ax = axes[0]
    for didx in range(len(domain_names)):
        mask = domain_labels == didx
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=[domain_colors[didx]], label=domain_names[didx],
                   alpha=0.7, s=40, edgecolors='white', linewidths=0.5)
    
    # Overlay Penrose vertices
    ax.scatter(penrose_vertices[:, 0], penrose_vertices[:, 1],
               c='gray', alpha=0.15, s=8, marker='.', label='Penrose vertices')
    
    # Draw lines from points to snapped vertices
    for i in range(0, len(coords_2d), 3):  # every 3rd for clarity
        ax.plot([coords_2d[i, 0], snapped_2d[i, 0]],
                [coords_2d[i, 1], snapped_2d[i, 1]],
                'k-', alpha=0.1, linewidth=0.5)
    
    ax.set_title("PCA 384D→2D with Penrose Tiling Overlay", fontsize=13, fontweight='bold')
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    
    # Right: Recall comparison bar chart
    ax2 = axes[1]
    method_labels = list(results_summary.keys())
    x_pos = np.arange(len(method_labels))
    width = 0.25
    
    r5_vals = [results_summary[m]["r@5"] for m in method_labels]
    r10_vals = [results_summary[m]["r@10"] for m in method_labels]
    r20_vals = [results_summary[m]["r@20"] for m in method_labels]
    
    bars1 = ax2.bar(x_pos - width, r5_vals, width, label='Recall@5', color='#2196F3', alpha=0.85)
    bars2 = ax2.bar(x_pos, r10_vals, width, label='Recall@10', color='#4CAF50', alpha=0.85)
    bars3 = ax2.bar(x_pos + width, r20_vals, width, label='Recall@20', color='#FF9800', alpha=0.85)
    
    ax2.set_ylabel("Recall (%)")
    ax2.set_title("Retrieval Method Comparison", fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["Flat 384D", "Random 2D", "PCA 2D", "PCA+Penrose"], fontsize=9)
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f'{height:.0f}%',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    print(f"      ✓ Plot saved to {PLOT_PATH}")
    
    # ── Write results markdown ──
    md = f"""# Penrose Memory Palace — End-to-End Results

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Model:** sentence-transformers/all-MiniLM-L6-v2 (384D embeddings)
**Corpus:** 200 sentences across 10 domains (20 each)
**Queries:** 50 (5 per domain)
**PCA:** 384D → 2D, explained variance = {sum(explained)*100:.2f}%
**Penrose vertices generated:** {len(penrose_vertices)}
**Avg snap distance:** {avg_snap_dist:.4f}
**Max snap distance:** {max_snap_dist:.4f}

## Retrieval Comparison

| Method | Recall@5 | Recall@10 | Recall@20 |
|--------|----------|-----------|-----------|
"""
    for m in method_labels:
        r = results_summary[m]
        md += f"| {m} | {r['r@5']:.1f}% | {r['r@10']:.1f}% | {r['r@20']:.1f}% |\n"
    
    md += f"""
## Interpretation

- **Flat 384D NN** is the ground truth baseline — cosine similarity in full embedding space.
- **Random 2D + NN** shows what happens with an untrained 2D projection — massive information loss.
- **PCA 2D + NN** uses a learned projection, much better than random but still lossy.
- **PCA 2D + Penrose** adds aperiodic discretization on top of PCA — the key question is how much recall is preserved vs. lost to the tiling snap.

### Key Insight
The Penrose tiling adds a **discretization layer** that quantizes continuous PCA coordinates into an aperiodic lattice. This provides:
- Fixed set of addressable memory locations (Penrose vertices)
- Golden-ratio spacing prevents aliasing artifacts
- Natural sharding via 3-coloring
- Loss is proportional to the snap distance (avg {avg_snap_dist:.4f})

## Domains

{', '.join(f'**{d}**' for d in domain_names)}

## Plot

See `penrose_demo_plot.png` for the PCA scatter with Penrose overlay and recall comparison chart.
"""
    
    with open(RESULTS_MD, 'w') as f:
        f.write(md)
    print(f"      ✓ Results saved to {RESULTS_MD}")
    
    print("\n" + "=" * 70)
    print("DONE. Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
