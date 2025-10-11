# Welcome to Alepo-Lab!

This repository is a collection of my practice projects, learning explorations, and code experiments. This space will serve as a living document of my growth and learning ventures as an engineer and researcher. Quick side projects, weekend projects, and experimentations will mostly populate this repository.

---

## Projects Catalog

Here is the curated list of projects within this repository. Each folder contains a detailed README.md with information on the project.

---

### Cloning a Single Project

If you only want to clone a single project without downloading the entire repository, use Git's `sparse-checkout` feature.

Replace `<project-folder>` with the name of the project directory you want to download:

```bash
# 1. Clone the repository without checking out any files
git clone --depth=1 --filter=blob:none --no-checkout [https://github.com/thealepo/alepo-lab.git](https://github.com/thealepo/alepo-lab.git)
cd alepo-lab

# 2. Specify the project folder you want
git sparse-checkout set <project-folder>
