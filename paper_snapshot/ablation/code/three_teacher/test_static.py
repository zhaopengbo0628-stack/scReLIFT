from __future__ import annotations

from multi_teacher import select_best_single_teacher, validate_teacher_matrix


def run():
    losses = {"A": {"A": .1, "B": .8, "C": .8},
              "B": {"A": .4, "B": .4, "C": .4},
              "C": {"A": .7, "B": .5, "C": .6}}
    validate_teacher_matrix({t: {s: .1 for s in "ABC"} for t in "ABC"}, tuple("ABC"))
    assert select_best_single_teacher(losses).selected_source == "B"
    print("MULTI_TEACHER_STATIC_TEST=PASS")


if __name__ == "__main__":
    run()
