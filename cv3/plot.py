"""
Vizualizace výkonu algoritmů pro vyhledávání vzorů v textu.
Spusťte: python plot.py (po spuštění main.go, které generuje CSV soubory)
"""

import csv
import matplotlib.pyplot as plt
import matplotlib
import os

# Nastavení fontu s podporou Unicode
matplotlib.rcParams['font.size'] = 11

def read_csv(filename):
    """Načte CSV soubor a vrátí seznam slovníků."""
    rows = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_scaling_by_text_length():
    """Graf 1: Počet porovnání vs délka textu."""
    rows = read_csv('scaling.csv')

    data = {}
    for row in rows:
        algo = row['Algorithm']
        if algo not in data:
            data[algo] = {'x': [], 'y': []}
        data[algo]['x'].append(int(row['TextLen']))
        data[algo]['y'].append(int(row['Comparisons']))

    markers = ['o', 's', '^']
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (algo, vals) in enumerate(data.items()):
        ax.plot(vals['x'], vals['y'], marker=markers[i], color=colors[i],
                label=algo, linewidth=2, markersize=7)

    ax.set_xlabel('Delka textu (znaky)', fontsize=12)
    ax.set_ylabel('Pocet porovnani znaku', fontsize=12)
    ax.set_title('Skalovani algoritmu dle delky textu\n(vzor "AGCT", DNA text)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain')
    plt.tight_layout()
    plt.savefig('graph_text_length.png', dpi=150)

def plot_scaling_by_pattern_length():
    """Graf 2: Počet porovnání vs délka vzoru."""
    rows = read_csv('pattern_scaling.csv')

    data = {}
    for row in rows:
        algo = row['Algorithm']
        if algo not in data:
            data[algo] = {'x': [], 'y': []}
        data[algo]['x'].append(int(row['PatternLen']))
        data[algo]['y'].append(int(row['Comparisons']))

    markers = ['o', 's', '^']
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (algo, vals) in enumerate(data.items()):
        ax.plot(vals['x'], vals['y'], marker=markers[i], color=colors[i],
                label=algo, linewidth=2, markersize=7)

    ax.set_xlabel('Delka vzoru (znaky)', fontsize=12)
    ax.set_ylabel('Pocet porovnani znaku', fontsize=12)
    ax.set_title('Skalovani algoritmu dle delky vzoru\n(DNA text 100 000 znaku)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain')
    plt.tight_layout()
    plt.savefig('graph_pattern_length.png', dpi=150)

def plot_comparison_bar_chart():
    """Graf 3: Sloupcový graf porovnání algoritmů pro různé typy textů."""
    rows = read_csv('results.csv')

    # Seskupení dle (TextType, Pattern)
    groups = {}
    for row in rows:
        key = f"{row['TextType']}\n\"{row['Pattern']}\""
        if key not in groups:
            groups[key] = {}
        groups[key][row['Algorithm']] = int(row['Comparisons'])

    labels = list(groups.keys())
    algo_names = ['Brute Force', 'KMP', 'BMH (Horspool)']
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, algo in enumerate(algo_names):
        vals = [groups[label].get(algo, 0) for label in labels]
        bars = ax.bar([xi + i * width for xi in x], vals, width,
                      label=algo, color=colors[i], alpha=0.85)
        # Přidání hodnot nad sloupce
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:,}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel('Typ textu / vzor', fontsize=12)
    ax.set_ylabel('Pocet porovnani znaku', fontsize=12)
    ax.set_title('Porovnani algoritmu na ruznych typech textu a vzoru', fontsize=13)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(labels, fontsize=8, ha='center')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('graph_comparison.png', dpi=150)


if __name__ == '__main__':
    # Změna pracovního adresáře na adresář skriptu
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    plot_scaling_by_text_length()
    plot_scaling_by_pattern_length()
    plot_comparison_bar_chart()