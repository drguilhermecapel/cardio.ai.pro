"""
SCP-ECG Standardized Diagnostic Statements for CardioAI Pro
Based on SCP-ECG recommendations for automated ECG interpretation
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass
class SCPCondition:
    """SCP-ECG diagnostic condition definition"""
    code: str
    name: str
    category: str
    icd10_codes: list[str]
    sensitivity_target: float
    specificity_target: float
    clinical_urgency: str
    description: str

class SCPCategory(str, Enum):
    """SCP-ECG diagnostic categories"""
    NORMAL = "normal"
    ARRHYTHMIA = "arrhythmia"
    CONDUCTION_DISORDER = "conduction_disorder"
    ISCHEMIA = "ischemia"
    HYPERTROPHY = "hypertrophy"
    AXIS_DEVIATION = "axis_deviation"
    REPOLARIZATION = "repolarization"
    OTHER = "other"

SCP_ECG_CONDITIONS = {
    "NORM": SCPCondition(
        code="NORM",
        name="Normal ECG",
        category=SCPCategory.NORMAL,
        icd10_codes=["Z03.89"],
        sensitivity_target=0.99,
        specificity_target=0.95,
        clinical_urgency="low",
        description="Normal electrocardiogram within normal limits"
    ),

    "AFIB": SCPCondition(
        code="AFIB",
        name="Atrial Fibrillation",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I48.0", "I48.1", "I48.2"],
        sensitivity_target=0.95,
        specificity_target=0.90,
        clinical_urgency="high",
        description="Atrial fibrillation with irregular RR intervals and absent P waves"
    ),

    "AFLT": SCPCondition(
        code="AFLT",
        name="Atrial Flutter",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I48.3", "I48.4"],
        sensitivity_target=0.95,
        specificity_target=0.90,
        clinical_urgency="high",
        description="Atrial flutter with regular atrial rate 250-350 bpm and sawtooth pattern"
    ),

    "SVTAC": SCPCondition(
        code="SVTAC",
        name="Supraventricular Tachycardia",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I47.1"],
        sensitivity_target=0.95,
        specificity_target=0.90,
        clinical_urgency="high",
        description="Supraventricular tachycardia with narrow QRS complexes"
    ),

    "VTAC": SCPCondition(
        code="VTAC",
        name="Ventricular Tachycardia",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I47.2"],
        sensitivity_target=0.98,
        specificity_target=0.95,
        clinical_urgency="critical",
        description="Ventricular tachycardia with wide QRS complexes >120ms"
    ),

    "VFIB": SCPCondition(
        code="VFIB",
        name="Ventricular Fibrillation",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.01"],
        sensitivity_target=0.99,
        specificity_target=0.98,
        clinical_urgency="critical",
        description="Ventricular fibrillation with chaotic irregular waveforms"
    ),

    "PVC": SCPCondition(
        code="PVC",
        name="Premature Ventricular Contractions",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.3"],
        sensitivity_target=0.90,
        specificity_target=0.85,
        clinical_urgency="medium",
        description="Premature ventricular contractions with wide QRS morphology"
    ),

    "PAC": SCPCondition(
        code="PAC",
        name="Premature Atrial Contractions",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.1"],
        sensitivity_target=0.85,
        specificity_target=0.80,
        clinical_urgency="low",
        description="Premature atrial contractions with abnormal P wave morphology"
    ),

    "BIGEMINY": SCPCondition(
        code="BIGEMINY",
        name="Ventricular Bigeminy",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.3"],
        sensitivity_target=0.90,
        specificity_target=0.85,
        clinical_urgency="medium",
        description="Ventricular bigeminy with alternating normal and premature beats"
    ),

    "TRIGEMINY": SCPCondition(
        code="TRIGEMINY",
        name="Ventricular Trigeminy",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.3"],
        sensitivity_target=0.90,
        specificity_target=0.85,
        clinical_urgency="medium",
        description="Ventricular trigeminy with every third beat being premature"
    ),

    "AVB1": SCPCondition(
        code="AVB1",
        name="First Degree AV Block",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I44.0"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="low",
        description="First degree AV block with PR interval >200ms"
    ),

    "AVB2M1": SCPCondition(
        code="AVB2M1",
        name="Second Degree AV Block Mobitz I",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I44.1"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="medium",
        description="Second degree AV block Mobitz I with progressive PR prolongation"
    ),

    "AVB2M2": SCPCondition(
        code="AVB2M2",
        name="Second Degree AV Block Mobitz II",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I44.1"],
        sensitivity_target=0.95,
        specificity_target=0.95,
        clinical_urgency="high",
        description="Second degree AV block Mobitz II with fixed PR intervals and dropped beats"
    ),

    "AVB3": SCPCondition(
        code="AVB3",
        name="Third Degree AV Block",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I44.2"],
        sensitivity_target=0.98,
        specificity_target=0.95,
        clinical_urgency="critical",
        description="Complete heart block with AV dissociation"
    ),

    "RBBB": SCPCondition(
        code="RBBB",
        name="Right Bundle Branch Block",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I45.0"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="low",
        description="Right bundle branch block with QRS >120ms and RSR' in V1"
    ),

    "LBBB": SCPCondition(
        code="LBBB",
        name="Left Bundle Branch Block",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I45.2"],
        sensitivity_target=0.95,
        specificity_target=0.95,
        clinical_urgency="medium",
        description="Left bundle branch block with QRS >120ms and broad R in V6"
    ),

    "LAFB": SCPCondition(
        code="LAFB",
        name="Left Anterior Fascicular Block",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I44.4"],
        sensitivity_target=0.85,
        specificity_target=0.90,
        clinical_urgency="low",
        description="Left anterior fascicular block with left axis deviation"
    ),

    "LPFB": SCPCondition(
        code="LPFB",
        name="Left Posterior Fascicular Block",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I44.5"],
        sensitivity_target=0.85,
        specificity_target=0.90,
        clinical_urgency="low",
        description="Left posterior fascicular block with right axis deviation"
    ),

    "WPW": SCPCondition(
        code="WPW",
        name="Wolff-Parkinson-White Syndrome",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I45.6"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="medium",
        description="WPW syndrome with delta waves and short PR interval"
    ),

    "STEMI": SCPCondition(
        code="STEMI",
        name="ST Elevation Myocardial Infarction",
        category=SCPCategory.ISCHEMIA,
        icd10_codes=["I21.0", "I21.1", "I21.2", "I21.3"],
        sensitivity_target=0.98,
        specificity_target=0.90,
        clinical_urgency="critical",
        description="ST elevation myocardial infarction with significant ST elevation"
    ),

    "NSTEMI": SCPCondition(
        code="NSTEMI",
        name="Non-ST Elevation Myocardial Infarction",
        category=SCPCategory.ISCHEMIA,
        icd10_codes=["I21.4"],
        sensitivity_target=0.90,
        specificity_target=0.85,
        clinical_urgency="critical",
        description="Non-ST elevation myocardial infarction with ST depression or T wave changes"
    ),

    "UAP": SCPCondition(
        code="UAP",
        name="Unstable Angina Pectoris",
        category=SCPCategory.ISCHEMIA,
        icd10_codes=["I20.0"],
        sensitivity_target=0.85,
        specificity_target=0.80,
        clinical_urgency="high",
        description="Unstable angina with dynamic ST-T changes"
    ),

    "ISCHEMIA": SCPCondition(
        code="ISCHEMIA",
        name="Myocardial Ischemia",
        category=SCPCategory.ISCHEMIA,
        icd10_codes=["I25.9"],
        sensitivity_target=0.85,
        specificity_target=0.80,
        clinical_urgency="medium",
        description="Myocardial ischemia with ST depression or T wave inversion"
    ),

    "QWAVE": SCPCondition(
        code="QWAVE",
        name="Pathological Q Waves",
        category=SCPCategory.ISCHEMIA,
        icd10_codes=["I25.2"],
        sensitivity_target=0.80,
        specificity_target=0.85,
        clinical_urgency="medium",
        description="Pathological Q waves suggesting old myocardial infarction"
    ),

    "LVH": SCPCondition(
        code="LVH",
        name="Left Ventricular Hypertrophy",
        category=SCPCategory.HYPERTROPHY,
        icd10_codes=["I51.7"],
        sensitivity_target=0.80,
        specificity_target=0.90,
        clinical_urgency="medium",
        description="Left ventricular hypertrophy by voltage criteria"
    ),

    "RVH": SCPCondition(
        code="RVH",
        name="Right Ventricular Hypertrophy",
        category=SCPCategory.HYPERTROPHY,
        icd10_codes=["I51.8"],
        sensitivity_target=0.75,
        specificity_target=0.90,
        clinical_urgency="medium",
        description="Right ventricular hypertrophy with tall R in V1"
    ),

    "LAE": SCPCondition(
        code="LAE",
        name="Left Atrial Enlargement",
        category=SCPCategory.HYPERTROPHY,
        icd10_codes=["I51.7"],
        sensitivity_target=0.70,
        specificity_target=0.85,
        clinical_urgency="low",
        description="Left atrial enlargement with P wave abnormalities"
    ),

    "RAE": SCPCondition(
        code="RAE",
        name="Right Atrial Enlargement",
        category=SCPCategory.HYPERTROPHY,
        icd10_codes=["I51.8"],
        sensitivity_target=0.70,
        specificity_target=0.85,
        clinical_urgency="low",
        description="Right atrial enlargement with tall P waves in II"
    ),

    "LAD": SCPCondition(
        code="LAD",
        name="Left Axis Deviation",
        category=SCPCategory.AXIS_DEVIATION,
        icd10_codes=["R94.31"],
        sensitivity_target=0.85,
        specificity_target=0.90,
        clinical_urgency="low",
        description="Left axis deviation with QRS axis -30° to -90°"
    ),

    "RAD": SCPCondition(
        code="RAD",
        name="Right Axis Deviation",
        category=SCPCategory.AXIS_DEVIATION,
        icd10_codes=["R94.31"],
        sensitivity_target=0.85,
        specificity_target=0.90,
        clinical_urgency="low",
        description="Right axis deviation with QRS axis +90° to +180°"
    ),

    "EAD": SCPCondition(
        code="EAD",
        name="Extreme Axis Deviation",
        category=SCPCategory.AXIS_DEVIATION,
        icd10_codes=["R94.31"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="medium",
        description="Extreme axis deviation with QRS axis -90° to +180°"
    ),

    "LQTS": SCPCondition(
        code="LQTS",
        name="Long QT Syndrome",
        category=SCPCategory.REPOLARIZATION,
        icd10_codes=["I45.81"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="high",
        description="Long QT syndrome with corrected QT >450ms (men) or >460ms (women)"
    ),

    "SQTS": SCPCondition(
        code="SQTS",
        name="Short QT Syndrome",
        category=SCPCategory.REPOLARIZATION,
        icd10_codes=["I45.89"],
        sensitivity_target=0.85,
        specificity_target=0.95,
        clinical_urgency="high",
        description="Short QT syndrome with corrected QT <350ms"
    ),

    "TWI": SCPCondition(
        code="TWI",
        name="T Wave Inversion",
        category=SCPCategory.REPOLARIZATION,
        icd10_codes=["R94.31"],
        sensitivity_target=0.80,
        specificity_target=0.75,
        clinical_urgency="medium",
        description="T wave inversion in multiple leads"
    ),

    "STTC": SCPCondition(
        code="STTC",
        name="ST-T Changes",
        category=SCPCategory.REPOLARIZATION,
        icd10_codes=["R94.31"],
        sensitivity_target=0.75,
        specificity_target=0.70,
        clinical_urgency="medium",
        description="Non-specific ST-T wave changes"
    ),

    "EARLY_REPOL": SCPCondition(
        code="EARLY_REPOL",
        name="Early Repolarization",
        category=SCPCategory.REPOLARIZATION,
        icd10_codes=["R94.31"],
        sensitivity_target=0.70,
        specificity_target=0.80,
        clinical_urgency="low",
        description="Early repolarization pattern with J point elevation"
    ),

    "BRADY": SCPCondition(
        code="BRADY",
        name="Sinus Bradycardia",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["R00.1"],
        sensitivity_target=0.90,
        specificity_target=0.85,
        clinical_urgency="medium",
        description="Sinus bradycardia with heart rate <60 bpm"
    ),

    "TACHY": SCPCondition(
        code="TACHY",
        name="Sinus Tachycardia",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["R00.0"],
        sensitivity_target=0.90,
        specificity_target=0.85,
        clinical_urgency="medium",
        description="Sinus tachycardia with heart rate >100 bpm"
    ),

    "PACE": SCPCondition(
        code="PACE",
        name="Paced Rhythm",
        category=SCPCategory.OTHER,
        icd10_codes=["Z95.0"],
        sensitivity_target=0.95,
        specificity_target=0.98,
        clinical_urgency="low",
        description="Paced rhythm with pacing spikes"
    ),

    "ARTIFACT": SCPCondition(
        code="ARTIFACT",
        name="Artifact",
        category=SCPCategory.OTHER,
        icd10_codes=["R94.31"],
        sensitivity_target=0.80,
        specificity_target=0.70,
        clinical_urgency="low",
        description="ECG artifact affecting interpretation"
    ),

    "LOW_VOLT": SCPCondition(
        code="LOW_VOLT",
        name="Low Voltage",
        category=SCPCategory.OTHER,
        icd10_codes=["R94.31"],
        sensitivity_target=0.75,
        specificity_target=0.80,
        clinical_urgency="low",
        description="Low voltage QRS complexes in limb leads"
    ),

    "POOR_PROG": SCPCondition(
        code="POOR_PROG",
        name="Poor R Wave Progression",
        category=SCPCategory.OTHER,
        icd10_codes=["R94.31"],
        sensitivity_target=0.70,
        specificity_target=0.75,
        clinical_urgency="low",
        description="Poor R wave progression in precordial leads"
    ),

    "JUNCTIONAL": SCPCondition(
        code="JUNCTIONAL",
        name="Junctional Rhythm",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.2"],
        sensitivity_target=0.85,
        specificity_target=0.90,
        clinical_urgency="medium",
        description="Junctional rhythm with absent or inverted P waves"
    ),

    "ESCAPE": SCPCondition(
        code="ESCAPE",
        name="Escape Rhythm",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.5"],
        sensitivity_target=0.80,
        specificity_target=0.85,
        clinical_urgency="medium",
        description="Escape rhythm with slow rate and wide QRS"
    ),

    "MULTIFOCAL_AT": SCPCondition(
        code="MULTIFOCAL_AT",
        name="Multifocal Atrial Tachycardia",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I47.1"],
        sensitivity_target=0.85,
        specificity_target=0.90,
        clinical_urgency="high",
        description="Multifocal atrial tachycardia with varying P wave morphology"
    ),

    "BIFASCICULAR": SCPCondition(
        code="BIFASCICULAR",
        name="Bifascicular Block",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I45.2"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="medium",
        description="Bifascicular block with RBBB and left fascicular block"
    ),

    "TRIFASCICULAR": SCPCondition(
        code="TRIFASCICULAR",
        name="Trifascicular Block",
        category=SCPCategory.CONDUCTION_DISORDER,
        icd10_codes=["I45.3"],
        sensitivity_target=0.95,
        specificity_target=0.95,
        clinical_urgency="high",
        description="Trifascicular block with bifascicular block and first degree AV block"
    ),

    "HYPERACUTE_T": SCPCondition(
        code="HYPERACUTE_T",
        name="Hyperacute T Waves",
        category=SCPCategory.ISCHEMIA,
        icd10_codes=["I21.9"],
        sensitivity_target=0.85,
        specificity_target=0.80,
        clinical_urgency="critical",
        description="Hyperacute T waves suggesting acute myocardial infarction"
    ),

    "WELLENS": SCPCondition(
        code="WELLENS",
        name="Wellens Syndrome",
        category=SCPCategory.ISCHEMIA,
        icd10_codes=["I20.0"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="critical",
        description="Wellens syndrome with biphasic T waves in V2-V3"
    ),

    "BRUGADA": SCPCondition(
        code="BRUGADA",
        name="Brugada Pattern",
        category=SCPCategory.REPOLARIZATION,
        icd10_codes=["I42.8"],
        sensitivity_target=0.95,
        specificity_target=0.98,
        clinical_urgency="critical",
        description="Brugada pattern with coved ST elevation in V1-V3"
    ),

    "ARVC": SCPCondition(
        code="ARVC",
        name="Arrhythmogenic Right Ventricular Cardiomyopathy",
        category=SCPCategory.OTHER,
        icd10_codes=["I42.8"],
        sensitivity_target=0.80,
        specificity_target=0.95,
        clinical_urgency="high",
        description="ARVC with epsilon waves and T wave inversion in V1-V3"
    ),

    "HYPERKALEMIA": SCPCondition(
        code="HYPERKALEMIA",
        name="Hyperkalemia",
        category=SCPCategory.OTHER,
        icd10_codes=["E87.5"],
        sensitivity_target=0.85,
        specificity_target=0.90,
        clinical_urgency="high",
        description="Hyperkalemia with peaked T waves and wide QRS"
    ),

    "HYPOKALEMIA": SCPCondition(
        code="HYPOKALEMIA",
        name="Hypokalemia",
        category=SCPCategory.OTHER,
        icd10_codes=["E87.6"],
        sensitivity_target=0.80,
        specificity_target=0.85,
        clinical_urgency="medium",
        description="Hypokalemia with U waves and ST depression"
    ),

    "DIGITALIS": SCPCondition(
        code="DIGITALIS",
        name="Digitalis Effect",
        category=SCPCategory.OTHER,
        icd10_codes=["T46.0X5A"],
        sensitivity_target=0.75,
        specificity_target=0.80,
        clinical_urgency="medium",
        description="Digitalis effect with sagging ST depression"
    ),

    "PERICARDITIS": SCPCondition(
        code="PERICARDITIS",
        name="Pericarditis",
        category=SCPCategory.OTHER,
        icd10_codes=["I30.9"],
        sensitivity_target=0.80,
        specificity_target=0.85,
        clinical_urgency="medium",
        description="Pericarditis with diffuse ST elevation and PR depression"
    ),

    "PULM_EMBOLISM": SCPCondition(
        code="PULM_EMBOLISM",
        name="Pulmonary Embolism",
        category=SCPCategory.OTHER,
        icd10_codes=["I26.9"],
        sensitivity_target=0.70,
        specificity_target=0.80,
        clinical_urgency="critical",
        description="Pulmonary embolism with S1Q3T3 pattern or right heart strain"
    ),

    "OSBORN_WAVE": SCPCondition(
        code="OSBORN_WAVE",
        name="Osborn Wave (Hypothermia)",
        category=SCPCategory.OTHER,
        icd10_codes=["T68"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="critical",
        description="Osborn waves (J waves) suggesting hypothermia"
    ),

    "EPSILON_WAVE": SCPCondition(
        code="EPSILON_WAVE",
        name="Epsilon Wave",
        category=SCPCategory.OTHER,
        icd10_codes=["I42.8"],
        sensitivity_target=0.85,
        specificity_target=0.95,
        clinical_urgency="high",
        description="Epsilon waves in V1-V3 suggesting ARVC"
    ),

    "DEXTROCARDIA": SCPCondition(
        code="DEXTROCARDIA",
        name="Dextrocardia",
        category=SCPCategory.OTHER,
        icd10_codes=["Q24.0"],
        sensitivity_target=0.95,
        specificity_target=0.98,
        clinical_urgency="low",
        description="Dextrocardia with inverted P waves and QRS in lead I"
    ),

    "LEAD_REVERSAL": SCPCondition(
        code="LEAD_REVERSAL",
        name="Lead Reversal",
        category=SCPCategory.OTHER,
        icd10_codes=["R94.31"],
        sensitivity_target=0.90,
        specificity_target=0.85,
        clinical_urgency="low",
        description="Lead reversal with abnormal axis and morphology"
    ),

    "NONSPECIFIC": SCPCondition(
        code="NONSPECIFIC",
        name="Nonspecific Abnormality",
        category=SCPCategory.OTHER,
        icd10_codes=["R94.31"],
        sensitivity_target=0.70,
        specificity_target=0.60,
        clinical_urgency="low",
        description="Nonspecific ECG abnormality requiring clinical correlation"
    ),

    "SINUS_ARRHYTHMIA": SCPCondition(
        code="SINUS_ARRHYTHMIA",
        name="Sinus Arrhythmia",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.8"],
        sensitivity_target=0.80,
        specificity_target=0.85,
        clinical_urgency="low",
        description="Sinus arrhythmia with respiratory variation in RR intervals"
    ),

    "WANDERING_PACEMAKER": SCPCondition(
        code="WANDERING_PACEMAKER",
        name="Wandering Atrial Pacemaker",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.8"],
        sensitivity_target=0.75,
        specificity_target=0.85,
        clinical_urgency="low",
        description="Wandering atrial pacemaker with varying P wave morphology"
    ),

    "ECTOPIC_ATRIAL": SCPCondition(
        code="ECTOPIC_ATRIAL",
        name="Ectopic Atrial Rhythm",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.1"],
        sensitivity_target=0.80,
        specificity_target=0.85,
        clinical_urgency="low",
        description="Ectopic atrial rhythm with abnormal P wave axis"
    ),

    "ACCELERATED_JUNCTIONAL": SCPCondition(
        code="ACCELERATED_JUNCTIONAL",
        name="Accelerated Junctional Rhythm",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.2"],
        sensitivity_target=0.80,
        specificity_target=0.90,
        clinical_urgency="medium",
        description="Accelerated junctional rhythm with rate 60-100 bpm"
    ),

    "ACCELERATED_VENTRICULAR": SCPCondition(
        code="ACCELERATED_VENTRICULAR",
        name="Accelerated Ventricular Rhythm",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.3"],
        sensitivity_target=0.85,
        specificity_target=0.90,
        clinical_urgency="medium",
        description="Accelerated ventricular rhythm with rate 50-100 bpm"
    ),

    "FUSION_BEATS": SCPCondition(
        code="FUSION_BEATS",
        name="Fusion Beats",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.9"],
        sensitivity_target=0.70,
        specificity_target=0.85,
        clinical_urgency="low",
        description="Fusion beats with combined supraventricular and ventricular activation"
    ),

    "ATRIAL_STANDSTILL": SCPCondition(
        code="ATRIAL_STANDSTILL",
        name="Atrial Standstill",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I49.5"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="high",
        description="Atrial standstill with absent P waves and junctional escape rhythm"
    ),

    "VENTRICULAR_STANDSTILL": SCPCondition(
        code="VENTRICULAR_STANDSTILL",
        name="Ventricular Standstill",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I46.9"],
        sensitivity_target=0.95,
        specificity_target=0.98,
        clinical_urgency="critical",
        description="Ventricular standstill with absent QRS complexes"
    ),

    "TORSADES_DE_POINTES": SCPCondition(
        code="TORSADES_DE_POINTES",
        name="Torsades de Pointes",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I47.2"],
        sensitivity_target=0.95,
        specificity_target=0.98,
        clinical_urgency="critical",
        description="Torsades de pointes with polymorphic VT and long QT"
    ),

    "BIDIRECTIONAL_VT": SCPCondition(
        code="BIDIRECTIONAL_VT",
        name="Bidirectional Ventricular Tachycardia",
        category=SCPCategory.ARRHYTHMIA,
        icd10_codes=["I47.2"],
        sensitivity_target=0.90,
        specificity_target=0.95,
        clinical_urgency="critical",
        description="Bidirectional VT with alternating QRS axis suggesting digitalis toxicity"
    )
}

HIERARCHICAL_STRUCTURE = {
    "level_1": {
        "normal": ["NORM"],
        "abnormal": [code for code in SCP_ECG_CONDITIONS.keys() if code != "NORM"]
    },
    "level_2": {
        SCPCategory.NORMAL: ["NORM"],
        SCPCategory.ARRHYTHMIA: [
            "AFIB", "AFLT", "SVTAC", "VTAC", "VFIB", "PVC", "PAC",
            "BIGEMINY", "TRIGEMINY", "BRADY", "TACHY", "JUNCTIONAL",
            "ESCAPE", "MULTIFOCAL_AT", "SINUS_ARRHYTHMIA", "WANDERING_PACEMAKER",
            "ECTOPIC_ATRIAL", "ACCELERATED_JUNCTIONAL", "ACCELERATED_VENTRICULAR", "FUSION_BEATS"
        ],
        SCPCategory.CONDUCTION_DISORDER: [
            "AVB1", "AVB2M1", "AVB2M2", "AVB3", "RBBB", "LBBB",
            "LAFB", "LPFB", "WPW", "BIFASCICULAR", "TRIFASCICULAR"
        ],
        SCPCategory.ISCHEMIA: [
            "STEMI", "NSTEMI", "UAP", "ISCHEMIA", "QWAVE",
            "HYPERACUTE_T", "WELLENS"
        ],
        SCPCategory.HYPERTROPHY: ["LVH", "RVH", "LAE", "RAE"],
        SCPCategory.AXIS_DEVIATION: ["LAD", "RAD", "EAD"],
        SCPCategory.REPOLARIZATION: [
            "LQTS", "SQTS", "TWI", "STTC", "EARLY_REPOL", "BRUGADA"
        ],
        SCPCategory.OTHER: [
            "PACE", "ARTIFACT", "LOW_VOLT", "POOR_PROG", "ARVC",
            "HYPERKALEMIA", "HYPOKALEMIA", "DIGITALIS", "PERICARDITIS",
            "PULM_EMBOLISM", "OSBORN_WAVE", "EPSILON_WAVE", "DEXTROCARDIA",
            "LEAD_REVERSAL", "NONSPECIFIC"
        ]
    }
}

CLINICAL_URGENCY_MAPPING = {
    "critical": [
        "VTAC", "VFIB", "AVB3", "STEMI", "NSTEMI", "HYPERACUTE_T",
        "WELLENS", "BRUGADA", "PULM_EMBOLISM", "OSBORN_WAVE",
        "VENTRICULAR_STANDSTILL", "TORSADES_DE_POINTES", "BIDIRECTIONAL_VT"
    ],
    "high": [
        "AFIB", "AFLT", "SVTAC", "AVB2M2", "UAP", "LQTS", "SQTS",
        "MULTIFOCAL_AT", "TRIFASCICULAR", "ARVC", "HYPERKALEMIA", "ATRIAL_STANDSTILL"
    ],
    "medium": [
        "PVC", "BIGEMINY", "TRIGEMINY", "AVB2M1", "LBBB", "WPW",
        "ISCHEMIA", "QWAVE", "LVH", "RVH", "EAD", "TWI", "STTC",
        "BRADY", "TACHY", "JUNCTIONAL", "ESCAPE", "BIFASCICULAR",
        "HYPOKALEMIA", "DIGITALIS", "PERICARDITIS", "ACCELERATED_JUNCTIONAL", "ACCELERATED_VENTRICULAR"
    ],
    "low": [
        "NORM", "PAC", "AVB1", "RBBB", "LAFB", "LPFB", "LAE", "RAE",
        "LAD", "RAD", "EARLY_REPOL", "PACE", "ARTIFACT", "LOW_VOLT",
        "POOR_PROG", "DEXTROCARDIA", "LEAD_REVERSAL", "NONSPECIFIC",
        "SINUS_ARRHYTHMIA", "WANDERING_PACEMAKER", "ECTOPIC_ATRIAL", "FUSION_BEATS"
    ]
}

def get_condition_by_code(code: str) -> SCPCondition | None:
    """Get SCP condition by code"""
    return SCP_ECG_CONDITIONS.get(code)

def get_conditions_by_category(category: SCPCategory) -> list[SCPCondition]:
    """Get all conditions in a specific category"""
    return [
        condition for condition in SCP_ECG_CONDITIONS.values()
        if condition.category == category
    ]

def get_critical_conditions() -> list[SCPCondition]:
    """Get all critical conditions requiring immediate attention"""
    critical_codes = CLINICAL_URGENCY_MAPPING["critical"]
    return [SCP_ECG_CONDITIONS[code] for code in critical_codes if code in SCP_ECG_CONDITIONS]

def get_conditions_by_urgency(urgency: str) -> list[SCPCondition]:
    """Get all conditions by clinical urgency level"""
    urgency_codes = CLINICAL_URGENCY_MAPPING.get(urgency, [])
    return [SCP_ECG_CONDITIONS[code] for code in urgency_codes if code in SCP_ECG_CONDITIONS]

def validate_scp_conditions() -> dict[str, Any]:
    """Validate SCP-ECG conditions completeness and consistency"""
    validation_results = {
        "total_conditions": len(SCP_ECG_CONDITIONS),
        "target_conditions": 71,
        "categories_covered": len(set(c.category for c in SCP_ECG_CONDITIONS.values())),
        "urgency_levels": len(set(c.clinical_urgency for c in SCP_ECG_CONDITIONS.values())),
        "conditions_with_icd10": len([c for c in SCP_ECG_CONDITIONS.values() if c.icd10_codes]),
        "high_sensitivity_conditions": len([c for c in SCP_ECG_CONDITIONS.values() if c.sensitivity_target >= 0.95]),
        "validation_passed": len(SCP_ECG_CONDITIONS) == 71
    }

    return validation_results
