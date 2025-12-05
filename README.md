# UOE Submissions
Dedicated to my submissions & work whilst pursuing my [MSc in Mathematical Finance](https://www.exeter.ac.uk/study/postgraduate/courses/mathematics/finmathsmsc) from the mathematics department at the University of Exeter, 2025-2026.

This work is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en).

## Module Legend

| Code                                                                                                      | Name                                            | Type     | CW1 | CW2 | CW3 | Final Grade |
| --------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | -------- | --- | --- | --- | ----------- |
| [MTHM002](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=MTHM002&ay=2025&sys=1) | Methods for Stochastics and Finance             | Core     | 90% |     |     |             |
| [MTHM003](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=MTHM003&ay=2025&sys=1) | Analysis and Computation for Finance            | Core     | 96% |     |     |             |
| [MTHM006](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=MTHM006&ay=2025&sys=1) | Mathematical Theory of Option Pricing           | Core     |     |     |     |             |
| [MTHM059](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=MTHM059&ay=2025&sys=1) | Case Studies in Mathematical Finance            | Core     |     |     |     |             |
| [MTHM060](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=MTHM060&ay=2025&sys=1) | Actuarial and Mathematical Finance Dissertation | Core     |     |     |     |             |
| [BEAM047](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=BEAM047&sys=0)         | Fundamentals of Financial Management            | Core     |     |     |     |             |
| [BEEM012](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=BEEM012&sys=0)         | Applied Econometrics 2                          | Elective |     |     |     |             |
| [BEAM035](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=BEAM035&sys=0)         | Derivatives Pricing                             | Elective |     |     |     |             |

## How To Use
To rebuild any submission PDFs, first clone the repo:

```bash
$> git clone https://github.com/RaSi96/uoe-submissions.git
$> cd uoe-submissions/
```

Then, to:
- Build everything
    ```bash
    $> make
    ```
- Clean everything
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

Compiled PDFs will appear in their specific `module-<code>/coursework-<N>/` folder. The root-level [`Makefile`](https://github.com/RaSi96/uoe-submissions/blob/dev/Makefile) also has commented instructions how to build submission PDFs.
