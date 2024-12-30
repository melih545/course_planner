import re
import pandas as pd
import streamlit as st
from pyscipopt import Model
import matplotlib.pyplot as plt
import matplotlib.patches as patches

GRADE_CREDITS = {1: 30, 2: 60, 3: 95}
SEMESTER_LST = ["Spring", "Summer", "Fall"]


def visualize_schedule(data, selected_courses, title="Schedule"):
    day_map = {"Pazartesi": 0, "Salı": 1, "Çarşamba": 2, "Perşembe": 3, "Cuma": 4}
    data["DayIndex"] = data["Days"].map(day_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    days = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma"]
    times = range(8, 22)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)
    ax.set_yticks(range(len(times)))
    ax.set_yticklabels([f"{t}:00" for t in times])
    for _, row in data.iterrows():
        if any(
            row["Code"] == course[0]
            and row["Days"] == course[3]
            and row["StartTimeFloat"] == course[4]
            for course in selected_courses
        ):
            try:
                if (
                    pd.isnull(row["Times"])
                    or "/" not in row["Times"]
                    and "490" not in row["Code"]
                ):
                    continue
                start_time, end_time = row["Times"].split("/")
                y = time_to_float(start_time) - 8
                height = time_to_float(end_time) - time_to_float(start_time)
                if height <= 0:
                    continue
                x = row["DayIndex"]
                label = f"{row['Code']}\n{row['CourseName']}\n{row['Instructors']}"
                rect = patches.Rectangle((x, y), 0.8, height, color="orange", alpha=0.6)
                ax.add_patch(rect)
                ax.text(
                    x + 0.4,
                    y + height / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    wrap=True,
                )
            except Exception as e:
                st.warning(f"Skipping invalid row: {row['Code']} (Error: {e})")
                continue

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 14)
    ax.set_xlabel("Days")
    ax.set_ylabel("Time")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


def to_semesters(term):
    if "Yaz" in term:
        return "Summer"
    elif "Güz" in term:
        return "Fall"
    elif "Bahar" in term:
        return "Spring"


def split_years(df):
    df["Year"] = df["Term"].str.extract(r"(\d{4} - \d{4})")
    df["Year"] = df["Year"].apply(
        lambda x: x.split("-")[0].strip() if "Güz" in x else x.split("-")[1].strip()
    )
    df["Term"] = df["Term"].str.extract(r"(Güz|Bahar|Yaz)")[0].apply(to_semesters)
    return df


def split_days_and_times(row):
    if (
        row["Days"] == "---"
        or row["Times"] == "/"
        or pd.isnull(row["Days"])
        or pd.isnull(row["Times"])
        or row["Instructors"] == "--"
    ):
        if "490" not in row["Code"]:
            return []
        row["Days"] = "Pazartesi"
        row["Times"] = "0000/0001"
    days = row["Days"].split()
    times = row["Times"].split()
    if len(days) != len(times):
        times = times * len(days)
    return [
        row.to_frame().T.assign(Days=day, Times=time).iloc[0]
        for day, time in zip(days, times)
    ]


def time_to_float(time_str):
    hours = int(time_str[:2])
    minutes = int(time_str[2:])
    return hours + minutes / 60


def generate_semesters(start_year, end_year):
    return [
        f"{term} {year}"
        for year in range(start_year, end_year + 1)
        for term in SEMESTER_LST
    ]


def check_conflict(row1, row2):
    return (
        row1["Days"] == row2["Days"]
        and row1["Term"] == row2["Term"]
        and (
            (
                row1["StartTimeFloat"] < row2["EndTimeFloat"]
                and row1["EndTimeFloat"] > row2["StartTimeFloat"]
            )
            or (
                row2["StartTimeFloat"] < row1["EndTimeFloat"]
                and row2["EndTimeFloat"] > row1["StartTimeFloat"]
            )
            or (row1["StartTimeFloat"] == row2["StartTimeFloat"])
            or (row1["EndTimeFloat"] == row2["EndTimeFloat"])
        )
    )


def prerequisites(pre):
    if type(pre) == float or not pre or "MIN" not in pre:
        return None
    matches = re.findall(
        r"\b(END \d+E|MAT \d+E|ATA \d+|TUR \d+|FIZ \d+EL|FIZ \d+E|KIM \d+EL|KIM \d+E)\b",
        pre,
    )
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return ", ".join(matches)


def calculate_take_probability(df):
    df["can_take"] = (df["Quota"] > df["Registered"]).astype(int)
    grouped_df = df.groupby(["Term", "CourseName"])["can_take"].mean().reset_index()
    grouped_df.rename(columns={"can_take": "probability_to_take"}, inplace=True)
    return grouped_df


# def update_take_probability(data, take_probability):


def calculate_open_rate(data):
    """
    Calculates the open rate for each lecture in specific semester types based on historical data.

    Parameters:
    data (pd.DataFrame): A DataFrame containing columns "Year", "Semester", and "Course Code".
                         - "Year": The year of the semester (e.g., 2015, 2016).
                         - "Semester": The semester type (e.g., Fall, Spring).
                         - "Course Code": The lecture identifier.

    Returns:
    pd.DataFrame: A DataFrame with the calculated open rate for each lecture in each semester type.
    """
    total_semester_counts = data.groupby("Term")["Year"].nunique().reset_index()
    total_semester_counts = total_semester_counts.rename(
        columns={"Year": "Total Semesters"}
    )

    course_counts = (
        data.groupby(["Term", "Code"])["Year"]
        .nunique()
        .reset_index()
        .rename(columns={"Year": "Course Count"})
    )

    course_counts = course_counts.merge(total_semester_counts, on="Term")
    course_counts["Open Rate"] = (
        course_counts["Course Count"] / course_counts["Total Semesters"]
    )

    return course_counts


def filter_recent_lectures(data, current_year):
    recent_years = list(range(current_year - 2, 2024))
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    recent_data = data[data["Year"].isin(recent_years)]
    recent_data = recent_data[
        (
            (recent_data["Year"].isin(recent_years))
            & (recent_data["Term"].isin(SEMESTER_LST))
        )
    ]
    recent_data["TermOrder"] = recent_data["Term"].map(
        {term: i for i, term in enumerate(SEMESTER_LST)}
    )
    recent_data["CourseName"] = recent_data["CourseName"].str.replace("&", "and")

    latest_codes = recent_data.sort_values(
        ["Year", "TermOrder"], ascending=[False, False]
    ).drop_duplicates(subset="CourseName", keep="first")[["CourseName", "Code"]]
    latest_durations = recent_data.sort_values(
        ["Year", "TermOrder"], ascending=[False, False]
    ).drop_duplicates(subset="Code", keep="first")[
        ["Code", "EndTimeFloat", "StartTimeFloat"]
    ]
    latest_durations["Duration"] = (
        latest_durations["EndTimeFloat"] - latest_durations["StartTimeFloat"]
    )
    duration_dict = latest_durations.set_index("Code")["Duration"].to_dict()
    recent_data = recent_data[
        recent_data["Code"].isin(latest_codes["Code"])
        & recent_data.apply(
            lambda x: abs(
                (x["EndTimeFloat"] - x["StartTimeFloat"])
                - duration_dict.get(x["Code"], 0)
                < 0.01
            ),
            axis=1,
        )
    ]

    valid_combinations = recent_data[["Code", "Instructors"]].drop_duplicates()

    filtered_data = pd.merge(
        data, valid_combinations, on=["Code", "Instructors"], how="inner"
    )

    return filtered_data.drop(columns=["TermOrder"], errors="ignore")


def process_uploaded_files(uploaded_file, necessary_lectures):
    data = pd.read_csv(uploaded_file)
    data.rename(
        columns={
            "Kod": "Code",
            "Ders": "CourseName",
            "Eğitmen(ler)": "Instructors",
            "Gün": "Days",
            "Saat": "Times",
            "Bina": "Building",
            "Kayıtlı": "Registered",
            "Kontenjan": "Quota",
            "Bölüm sınırlaması": "DepartmentRestrictions",
        },
        inplace=True,
    )
    filtered_data = data[data["DepartmentRestrictions"].str.contains("ENDE", na=False)]
    lect_dct = {
        row["Course Code"]: {
            "Credits": row["Credits"],
            "Prerequisites": row["Prerequisites"],
            "Grade": (float(row["Semester"].split(".")[0]) + 1) // 2,
            "Type": row["Type"],
        }
        for idx, row in necessary_lectures.iterrows()
    }
    filtered_data = filtered_data[
        filtered_data["Code"].isin(necessary_lectures["Course Code"])
    ]

    expanded_data = []
    filtered_data = split_years(filtered_data)
    for _, row in filtered_data.iterrows():
        row["Credits"] = lect_dct[row["Code"]]["Credits"]
        row["Prerequisites"] = prerequisites(lect_dct[row["Code"]]["Prerequisites"])
        row["Grade"] = lect_dct[row["Code"]]["Grade"]
        row["Type"] = lect_dct.get(row["Code"], {}).get("Type")
        expanded_data.extend(split_days_and_times(row))
    expanded_data = pd.DataFrame(expanded_data)

    expanded_data["StartTimeFloat"] = (
        expanded_data["Times"].str.split("/").str[0].apply(time_to_float)
    )
    expanded_data["EndTimeFloat"] = (
        expanded_data["Times"].str.split("/").str[1].apply(time_to_float)
    )
    expanded_data = filter_recent_lectures(expanded_data, 2023)
    expanded_data = expanded_data[expanded_data["EndTimeFloat"] <= 17.5]
    expanded_data["Quota"] = pd.to_numeric(expanded_data["Quota"], errors="coerce")
    expanded_data = expanded_data[
        (expanded_data["Quota"] >= 1)
        | (expanded_data["Code"].str.contains("490", na=False))
    ]
    open_rates = calculate_open_rate(expanded_data)
    expanded_data = expanded_data.merge(
        open_rates[["Term", "Code", "Open Rate"]], on=["Term", "Code"], how="left"
    )
    expanded_data = expanded_data.merge(
        calculate_take_probability(expanded_data), on=["Term", "CourseName"], how="left"
    )
    expanded_data = expanded_data.drop_duplicates(
        subset=["Code", "Term", "Days", "StartTimeFloat"], keep="first"
    ).reset_index(drop=True)
    latest_names = expanded_data.sort_values(
        by="Year", ascending=False
    ).drop_duplicates(subset=["Code"], keep="first")
    expanded_data = expanded_data.drop(columns=["CourseName"]).merge(
        latest_names[["Code", "CourseName"]], on="Code", how="left"
    )
    rows_to_keep = []
    rows_to_keep.extend(
        [
            idx
            for idx, row in expanded_data.iterrows()
            if not any(
                check_conflict(row, expanded_data.iloc[other_idx])
                for other_idx in rows_to_keep
                if row["Code"] == expanded_data.iloc[other_idx]["Code"]
                and row["Term"] == expanded_data.iloc[other_idx]["Term"]
                and row["Days"] == expanded_data.iloc[other_idx]["Days"]
            )
        ]
    )
    return expanded_data.iloc[rows_to_keep].reset_index(drop=True)


def main():
    st.title("Interactive Course Planner with Iterative Scheduling")

    uploaded_file = "all_lectures.csv"
    necessary_lectures = "course_plans.csv"

    if "remaining_data" not in st.session_state:
        st.session_state.remaining_data = None

    if "completed_courses" not in st.session_state:
        st.session_state.completed_courses = []

    if uploaded_file and necessary_lectures:
        necessary = pd.read_csv(necessary_lectures)
        if st.session_state.remaining_data is None:
            st.session_state.remaining_data = process_uploaded_files(
                uploaded_file, necessary
            )
        completed_courses = st.multiselect(
            "Select completed courses",
            options=list(necessary["Course Code"]),
        )
        col1, col2 = st.columns(2)

        with col1:
            temp_current_year = st.selectbox(
                "Select the current year",
                options=range(2024, 2035),
                index=0,
            )
        with col2:
            temp_current_semester = st.selectbox(
                "Select the current semester",
                options=SEMESTER_LST,
                index=0,
            )
        with col1:
            temp_last_year = st.selectbox(
                "Select the year you want to finish",
                options=range(temp_current_year, 2035),
                index=0,
            )
        with col2:
            temp_last_semester = st.selectbox(
                "Select the last semester you attended",
                options=SEMESTER_LST,
                index=0,
            )
        if "calculated" not in st.session_state:
            st.session_state.calculated = False

        if (
            "current_year" not in st.session_state
            or st.session_state.current_year != temp_current_year
        ) and st.session_state.calculated == False:
            st.session_state.current_year = temp_current_year
        if (
            "current_semester" not in st.session_state
            or st.session_state.current_semester != temp_current_semester
        ) and st.session_state.calculated == False:
            st.session_state.current_semester = temp_current_semester

        if (
            "last_year" not in st.session_state
            or st.session_state.last_year != temp_last_year
        ) and st.session_state.calculated == False:
            st.session_state.last_year = temp_last_year

        if (
            "last_semester" not in st.session_state
            or st.session_state.last_semester != temp_last_semester
        ) and st.session_state.calculated == False:
            st.session_state.last_semester = temp_last_semester

        if "path" not in st.session_state:
            st.session_state.path = {}
        if "updated" not in st.session_state:
            st.session_state.updated = True

        if st.button("Calculate", key="Calculate"):
            st.session_state.calculated = True

        if st.session_state.calculated:
            data = st.session_state.remaining_data
            st.write(st.session_state.remaining_data)
            if not st.session_state.completed_courses:
                st.session_state.completed_courses.extend(completed_courses)
            if "path_data" in st.session_state:
                for term, lectures in st.session_state.path.items():
                    st.header(f"The planned schedule for {term}")
                    visualize_schedule(
                        st.session_state.path_data,
                        lectures,
                        title=f"Selected Schedule for {term.replace('_', ' ')}",
                    )
            if st.session_state.updated:
                st.session_state.updated = False
                completed_credits = necessary[
                    necessary["Course Code"].isin(st.session_state.completed_courses)
                ]["Credits"].sum()
                current_grade_level = max(
                    [
                        grade
                        for grade, credits in GRADE_CREDITS.items()
                        if completed_credits >= credits
                    ],
                    default=1,
                )
                st.write(f"Current Grade Level: {current_grade_level}")
                years = list(
                    range(
                        st.session_state.current_year,
                        int(st.session_state.last_year + 1),
                    )
                )
                all_semesters = [
                    (year, term) for year in years for term in SEMESTER_LST
                ]
                st.header(
                    f"Planning for {st.session_state.current_year} {st.session_state.current_semester}"
                )

                model = Model("Course_Planner")

                # Ensure that each functional and analytic elective is taken at least 4 times
                course_vars = {
                    (
                        row["Code"],
                        year,
                        row["Term"],
                        row["Days"],
                        row["StartTimeFloat"],
                    ): model.addVar(
                        vtype="B",
                        name=f"Take_{row['Code']}_{year}_{row['Term']}_{row['Days']}_at_{row['StartTimeFloat']}",
                    )
                    for _, row in data.iterrows()
                    for year in years
                }
                for elective_type in data[data["Type"].str.contains("Elective")][
                    "Type"
                ].unique():
                    model.addCons(
                        sum(
                            course_vars[
                                (
                                    row["Code"],
                                    year,
                                    row["Term"],
                                    row["Days"],
                                    row["StartTimeFloat"],
                                )
                            ]
                            for _, row in data[data["Type"] == elective_type].iterrows()
                            for year in years
                        )
                        >= 4,
                        name=f"ElectiveCreditsConstraint_{elective_type.split(' ')[1]}",
                    )

                # Ensure that at least one ITB course is taken
                model.addCons(
                    sum(
                        course_vars[
                            (
                                row["Code"],
                                year,
                                row["Term"],
                                row["Days"],
                                row["StartTimeFloat"],
                            )
                        ]
                        for _, row in data[data["Type"] == "ITB"].iterrows()
                        for year in years
                    )
                    >= 1,
                    name="ITBConstraint",
                )

                # Ensure each course is scheduled exactly once
                unique_codes = data["Code"].unique()
                for course in unique_codes:
                    if course in st.session_state.completed_courses:
                        continue
                    constraints = []
                    terms = data.loc[data["Code"] == course, "Term"].unique()

                    for term in terms:
                        days = data.loc[
                            (data["Code"] == course) & (data["Term"] == term),
                            "Days",
                        ].unique()

                        for day in days:
                            times = data.loc[
                                (data["Code"] == course)
                                & (data["Term"] == term)
                                & (data["Days"] == day),
                                "StartTimeFloat",
                            ].unique()

                            for time in times:
                                constraints.extend(
                                    [
                                        course_vars[(course, year, term, day, time)]
                                        for year in years
                                    ]
                                )

                    course_type = data.loc[data["Code"] == course, "Type"].iloc[0]
                    is_elective = (
                        any(t in course_type for t in ["Elective", "ITB"])
                        if course_type
                        else False
                    )

                    if is_elective:
                        model.addCons(
                            sum(constraints) <= 1, name=f"UniqueCode_{course}"
                        )
                    else:
                        model.addCons(
                            sum(constraints) == 1, name=f"UniqueCode_{course}"
                        )

                # Check conflicts between courses
                for num, (_, course1) in enumerate(data.iterrows()):
                    for _, course2 in data.iloc[num:].iterrows():
                        if course1["Code"] != course2["Code"] and check_conflict(
                            course1, course2
                        ):
                            for year in years:
                                course1_var = course_vars[
                                    (
                                        course1["Code"],
                                        year,
                                        course1["Term"],
                                        course1["Days"],
                                        course1["StartTimeFloat"],
                                    )
                                ]
                                course2_var = course_vars[
                                    (
                                        course2["Code"],
                                        year,
                                        course2["Term"],
                                        course2["Days"],
                                        course2["StartTimeFloat"],
                                    )
                                ]

                                model.addCons(
                                    course1_var + course2_var <= 1,
                                    name=f"Conflict_{course1['Code']}_{course2['Code']}_{year}_{course1['Term']}_{course1['Days']}_{course1['StartTimeFloat']}_{course2['StartTimeFloat']}",
                                )

                # Precompute prerequisite mappings to reduce iterations
                prereq_map = {}
                for _, row in data.iterrows():
                    if pd.notnull(row["Prerequisites"]) and row["Prerequisites"]:
                        prereqs = row["Prerequisites"].split(", ")
                        prereq_map[row["Code"]] = [
                            pre
                            for pre in prereqs
                            if pre not in st.session_state.completed_courses
                        ]

                # Create aggregated variables for courses taken in a term
                term_vars = {
                    (course, year, term): model.addVar(
                        vtype="B",
                        name=f"Take_{course}_{year}_{term}_anytime",
                    )
                    for course in data["Code"].unique()
                    for year in years
                    for term in SEMESTER_LST
                }

                # Add constraints to link term_vars with the detailed course_vars
                for course in unique_codes:
                    for year in years:
                        for term in data[data["Code"] == course]["Term"].unique():
                            model.addCons(
                                sum(
                                    course_vars[(course, year, term, day, time)]
                                    for day in data[
                                        (data["Code"] == course)
                                        & (data["Term"] == term)
                                    ]["Days"].unique()
                                    for time in data[
                                        (data["Code"] == course)
                                        & (data["Term"] == term)
                                        & (data["Days"] == day)
                                    ]["StartTimeFloat"].unique()
                                )
                                == term_vars[(course, year, term)],
                                name=f"Link_{course}_{year}_{term}",
                            )

                # Add prerequisite constraints
                for course, prereqs in prereq_map.items():
                    for pre in prereqs:
                        if pre not in data["Code"].values:
                            continue
                        for year in years:
                            for term in data[data["Code"] == course]["Term"].unique():
                                prereq_terms = [
                                    (prereq_year, prereq_term)
                                    for prereq_year in years
                                    for prereq_term in data[data["Code"] == pre][
                                        "Term"
                                    ].unique()
                                    if (prereq_year < year)
                                    or (
                                        prereq_year == year
                                        and SEMESTER_LST.index(prereq_term)
                                        <= SEMESTER_LST.index(term)
                                    )
                                ]

                                if prereq_terms:
                                    valid_prereq_terms = [
                                        term_vars[(pre, prereq_year, prereq_term)]
                                        for prereq_year, prereq_term in prereq_terms
                                        if not (
                                            prereq_year == year and prereq_term == term
                                        )
                                    ]
                                    if valid_prereq_terms:
                                        model.addCons(
                                            term_vars[(course, year, term)]
                                            <= sum(valid_prereq_terms),
                                            name=f"Prerequisite_{course}_{pre}_{year}_{term}",
                                        )
                                    else:
                                        model.addCons(
                                            term_vars[(course, year, term)] == 0,
                                            name=f"Prerequisite_{course}_{pre}_{year}_{term}_Invalid",
                                        )

                semester_vars = {
                    (year, term): model.addVar(vtype="B", name=f"Use_{year}_{term}")
                    for year in years
                    for term in SEMESTER_LST
                }

                # Constraint: If a course is taken in a semester, that semester must be marked as used
                for year in years:
                    for term in SEMESTER_LST:
                        model.addCons(
                            sum(
                                course_vars[
                                    (
                                        row["Code"],
                                        year,
                                        term,
                                        row["Days"],
                                        row["StartTimeFloat"],
                                    )
                                ]
                                for _, row in data.iterrows()
                                if (
                                    row["Code"],
                                    year,
                                    term,
                                    row["Days"],
                                    row["StartTimeFloat"],
                                )
                                in course_vars
                            )
                            <= semester_vars[(year, term)] * 1000,
                            name=f"LinkCoursesToSemester_{year}_{term}",
                        )

                model.setObjective(
                    sum(
                        semester_vars[(year, term)]
                        for year in years
                        for term in SEMESTER_LST
                    ),
                    "minimize",
                )

                for i, (year, term) in enumerate(all_semesters[:-1]):
                    if term == "Summer":
                        year, term = all_semesters[i - 1]
                    next_year, next_term = all_semesters[i + 1]
                    model.addCons(
                        semester_vars[(year, term)]
                        >= semester_vars[(next_year, next_term)],
                        name=f"NoGap_{year}_{term}_to_{next_year}_{next_term}",
                    )

                paths = []
                solution_count = 0

                # Initial optimization to find the first solution
                model.optimize()
                while (
                    model.getStatus() in ["optimal", "feasible"] and solution_count < 5
                ):
                    sol = model.getBestSol()
                    current_schedule = [
                        [
                            row["Code"],
                            year,
                            row["Term"],
                            row["Days"],
                            row["StartTimeFloat"],
                        ]
                        for _, row in data.iterrows()
                        for year in years
                        if model.getSolVal(
                            sol,
                            course_vars[
                                (
                                    row["Code"],
                                    year,
                                    row["Term"],
                                    row["Days"],
                                    row["StartTimeFloat"],
                                )
                            ],
                        )
                        > 0.5
                    ]

                    if current_schedule not in paths:
                        paths.append(current_schedule)
                        solution_count += 1

                    model.freeTransform()
                    excluded_sol = [
                        course_vars[tuple(course)] for course in current_schedule
                    ]
                    model.addCons(sum(excluded_sol) <= len(current_schedule) - 1)

                    model.optimize()

                st.write(f"Total Solutions Found: {len(paths)}")
                st.session_state.unique_paths = paths
            for i, path in enumerate(st.session_state.unique_paths):
                term_path = [
                    course
                    for course in path
                    if course[1] == st.session_state.current_year
                    and course[2] == st.session_state.current_semester
                ]
                if term_path:
                    st.subheader(f"Option {i+1}")
                    itb_count = len(
                        [
                            course
                            for course in term_path
                            if data[data["Code"] == course[0]]["Type"].iloc[0] == "ITB"
                        ]
                    )
                    elective_analytic = len(
                        [
                            course
                            for course in term_path
                            if data[data["Code"] == course[0]]["Type"].iloc[0]
                            == "Elective Analytic"
                        ]
                    )
                    elective_functional = len(
                        [
                            course
                            for course in term_path
                            if data[data["Code"] == course[0]]["Type"].iloc[0]
                            == "Elective Functional"
                        ]
                    )
                    main_courses = len(
                        [
                            course
                            for course in term_path
                            if data[data["Code"] == course[0]]["Type"].iloc[0]
                            not in [
                                "ITB",
                                "Elective Analytic",
                                "Elective Functional",
                            ]
                        ]
                    )
                    st.subheader(f"The Course That Are Included in This Path:")
                    st.write(f"ITB Courses: {itb_count}")
                    st.write(f"Elective Analytic Courses: {elective_analytic}")
                    st.write(f"Elective Functional Courses: {elective_functional}")
                    st.write(f"Main Courses: {main_courses}")
                    visualize_schedule(data, term_path, title=f"Option {i+1} Schedule")
                else:
                    st.write(
                        f"Option {i+1} for {st.session_state.current_year} {st.session_state.current_semester} does not include any courses."
                    )

            selected_path = st.radio(
                "Select your preferred schedule:",
                [f"Option {i+1}" for i in range(len(st.session_state.unique_paths))],
                key=f"radio_{st.session_state.current_year}_{st.session_state.current_semester}",
            )
            button = st.button(
                label=f"Confirm {st.session_state.current_year} {st.session_state.current_semester}",
                key=f"confirm_{st.session_state.current_year}_{st.session_state.current_semester}",
            )

            if button:
                if not selected_path:
                    st.error("Please select a schedule before proceeding.")
                else:
                    selected_courses = [
                        course
                        for course in st.session_state.unique_paths[
                            int(selected_path.split(" ")[1]) - 1
                        ]
                        if course[1] == st.session_state.current_year
                        and course[2] == st.session_state.current_semester
                    ]

                    st.session_state.completed_courses.extend(
                        [course[0] for course in selected_courses]
                    )
                    st.session_state.remaining_data = data[
                        ~data["Code"].isin(st.session_state.completed_courses)
                    ].reset_index(drop=True)
                    if "path_data" not in st.session_state:
                        st.session_state.path_data = pd.DataFrame()
                    st.session_state.path_data = pd.concat(
                        [
                            st.session_state.path_data,
                            data[
                                data.apply(
                                    lambda row: any(
                                        row["Code"] == course[0]
                                        and row["Term"] == course[2]
                                        and row["Days"] == course[3]
                                        and row["StartTimeFloat"] == course[4]
                                        for course in selected_courses
                                    ),
                                    axis=1,
                                )
                            ],
                        ]
                    ).reset_index(drop=True)
                    st.session_state.path.update(
                        {
                            f"{st.session_state.current_year}_{st.session_state.current_semester}": selected_courses
                        }
                    )
                    if st.session_state.current_semester == "Fall":
                        st.session_state.current_year += 1
                    st.session_state.current_semester = SEMESTER_LST[
                        (SEMESTER_LST.index(st.session_state.current_semester) + 1)
                        % len(SEMESTER_LST)
                    ]
                    st.session_state.updated = True
                    st.rerun()


if __name__ == "__main__":
    main()
