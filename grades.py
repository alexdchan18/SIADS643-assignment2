import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm, t

def letter_to_points(letter_grade):
    """
    Converts a letter grade into its equivalent grade points.

    This function maps letter grades commonly used in educational institutions
    to a numerical value representing grade points. It covers grades from 'A+'
    through 'E', with 'A+' being the highest and 'E' being the lowest. Any
    letter grade not recognized is returned as NaN (Not a Number).

    Parameters
    ----------
    letter_grade : str
        A string representing the letter grade to be converted into grade points.

    Returns
    -------
    float
        The grade points corresponding to the input letter grade. Returns NaN if
        the letter grade is not recognized.
    """
    grade_mapping = {
        'A+': 4.3, 'A': 4.0, 'A-': 3.7,
        'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7,
        'D+': 1.3, 'D': 1.0, 'D-': 0.7,
        'E': 0.0
    }

    return grade_mapping.get(letter_grade, np.nan)

def grade_distribution(grade_data):
    """
    Visualizes the grade distribution for specified courses using grade data from a CSV file.

    This function reads grade data for six courses from a specified CSV file, converts letter grades to grade points,
    and plots the distribution of grades for each course. For one course (STATS250), it also overlays the plot with
    the distribution of a random sample of 100 grades using both a normal and a T-distribution.

    Parameters:
    - grade_data (str): Path to the CSV file containing grade data. The CSV should include columns for courses
      ('STATS250_grade', 'DATASCI306_grade', 'MATH217_grade', 'ENGLISH125_grade', 'ECON101_grade', 'EECS545_grade')
      with letter grades.

    The function converts letter grades to grade points based on a predefined scale, plots histograms for the grade
    distributions of the courses, and overlays these with the appropriate distribution curves. It generates a 3x2 subplot
    grid to visualize these distributions for the six courses.

    Note:
    - The function depends on pandas for data handling, matplotlib for plotting, and scipy for statistical functions.
    - `letter_to_points` function must be defined in the same script or notebook to convert letter grades to points.

    Outputs:
    - A matplotlib figure with a 3x2 grid of subplots, each showing the grade distribution for one of the six specified courses.
      For STATS250, the distribution of a sample of 100 grades is also shown with normal and T-distribution curves.
    """
    df = pd.read_csv(grade_data)
    
    # Add a new column for each course with _grade_points at the end
    for course in ['STATS250', 'DATASCI306', 'MATH217', 'ENGLISH125', 'ECON101', 'EECS545']:
        df[course + '_grade_points'] = df[course + '_grade'].apply(letter_to_points)
    
    # Create plot
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the array to make it easier to index
    courses = ['STATS250', 'DATASCI306', 'MATH217', 'ENGLISH125', 'ECON101', 'EECS545']
    
    # Iterate through courses and create graph
    for i, course in enumerate(courses):
        course_grades = df[course + '_grade_points'].dropna()
        n_students = len(course_grades)
        mean = course_grades.mean()
        std = course_grades.std()

        sns.histplot(course_grades, bins=30, kde=False, ax=axs[i], stat='density', legend=False)
        
        x_values = np.linspace(mean - 4*std, mean + 4*std, 200)
        axs[i].plot(x_values, norm.pdf(x_values, mean, std), label=f'Normal Dist, µ={mean:.2f}, σ={std:.2f}')
        
        if course == 'STATS250':
            sample_grades = course_grades.sample(100, random_state=42)
            sample_mean = sample_grades.mean()
            sample_std = sample_grades.std(ddof=1)
            axs[i].plot(x_values, t.pdf(x_values, df=len(sample_grades)-1, loc=sample_mean, scale=sample_std), label=f'T-Dist, µ={sample_mean:.2f}, σ={sample_std:.2f}')
        
        axs[i].legend(title=f"{course}, n={n_students}")
        axs[i].set_title(f"Grade Distribution for {course}")
    
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python grades.py <input_file>')
    else:
        grade_distribution(sys.argv[1])
