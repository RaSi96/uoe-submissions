# UOE Submissions
Dedicated to my submissions & work whilst pursuing my [MSc in Mathematical Finance](https://www.exeter.ac.uk/study/postgraduate/courses/mathematics/finmathsmsc) from the mathematics department at the University of Exeter, 2025-2026.

This work is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en).

## Module Legend

| Code                                                                                                      | Name                                            | Type     | CW1 | CW2 | CW3 | Final Grade |
| --------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | -------- | --- | --- | --- | ----------- |
| [MTHM002](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=MTHM002&ay=2025&sys=1) | Methods for Stochastics and Finance             | Core     | 90% |     | N/A |             |
| [MTHM003](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=MTHM003&ay=2025&sys=1) | Analysis and Computation for Finance            | Core     | 96% |     | N/A |             |
| [MTHM006](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=MTHM006&ay=2025&sys=1) | Mathematical Theory of Option Pricing           | Core     |     |     |     |             |
| [MTHM059](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=MTHM059&ay=2025&sys=1) | Case Studies in Mathematical Finance            | Core     | 91% |     |     |             |
| [MTHM060](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=MTHM060&ay=2025&sys=1) | Actuarial and Mathematical Finance Dissertation | Core     |     |     |     |             |
| [BEAM047](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=BEAM047&sys=0)         | Fundamentals of Financial Management            | Core     | N/A | N/A | N/A |             |
| [BEEM012](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=BEEM012&sys=0)         | Applied Econometrics 2                          | Elective |     |     |     |             |
| [BEAM035](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=BEAM035&sys=0)         | Derivatives Pricing                             | Elective | N/A | N/A | N/A |             |

## How To Use
To rebuild any submission PDFs, first clone the repo:

```bash
$> git clone https://github.com/RaSi96/uoe-submissions.git
$> cd uoe-submissions/
```

Then, to:
- Build everything:
    ```bash
    $> make
    ```
- Clean everything:
    ```bash
    $> make clean
    ```
- Build all coursework under a subject:
    ```bash
    $> make -C module-mthm003
    ```
- Clean all coursework under a subject:
    ```bash
    $> make -C module-mthm003 clean
    ```
- Build a single coursework:
    ```bash
    $> make -C module-mthm003/coursework-1
    ```
- Clean a single coursework:
    ```bash
    $> make -C module-mthm003/coursework-1 clean
    ```

You can also navigate to each (sub-)directory and run `make` from there if you prefer not to do it from root-level; it's the same. Compiled PDFs will appear in their specific `module-<code>/coursework-<N>/` folder. The root-level [`Makefile`](https://github.com/RaSi96/uoe-submissions/blob/dev/Makefile) also has commented instructions how to build submission PDFs.

## Statement of Academic Honesty
As mentioned, this repo is dedicated to hosting _my_ submissions to coursework questions asked of me, during my MSc in Mathematical Finance as a student of the University of Exeter from September 2025 to September 2026. All the work here is my original work, uploaded exactly as submitted, with citations and references aligned with the University's academic standards; shared for **educational, illustrative, and reference purposes only**. Note that I only upload courseworks _after_ they have been formally graded.

Just as I worked on all of this by myself, it is in the interest of all humanity and a reflection of but a modicum of common sense & integrity that future students also work on their coursework independently, and in line with their institution's academic code. This material is **not** to be adapted or submitted - either wholly, or in part - as part of any academic assessments elsewhere, as doing so would violate academic integrity principles. You, dear reader, may view, read, and learn from this material, but you may **not** republish, reuse for assessment, claim authorship, or plagiarise any part of this work. Any misuse is solely the responsibility of the individual who chooses to engage in it.

For a general overview on the University of Exeter's academic integrity philosophy, please [visit this webpage](https://www.exeter.ac.uk/students/facultycases/academicconductandpractice/).