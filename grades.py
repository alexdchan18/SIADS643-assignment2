import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm, t

def letter_to_points(letter_grade):
    grade_mapping = {
        'A+': 4.3, 'A': 4.0, 'A-': 3.7,
        'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7,
        'D+': 1.3, 'D': 1.0, 'D-': 0.7,
        'E': 0.0
    }
    return grade_mapping.get(letter_grade, np.nan)

def grade_distribution():
    df = pd.read_csv('assets/assets/class_grades.csv')
    
    for course in ['STATS250', 'DATASCI306', 'MATH217', 'ENGLISH125', 'ECON101', 'EECS545']:
        df[course + '_grade_points'] = df[course + '_grade'].apply(letter_to_points)
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the array to make it easier to index
    courses = ['STATS250', 'DATASCI306', 'MATH217', 'ENGLISH125', 'ECON101', 'EECS545']
    
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

grade_distribution()
