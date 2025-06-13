import logging
from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

logger = logging.getLogger(__name__)


class PatientAwareValidator:
    """Sistema de validação que previne data leakage entre pacientes"""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.validation_history: list[dict[str, Any]] = []

    def create_patient_aware_splits(self,
                                  X: NDArray[np.floating[Any]],
                                  y: NDArray[np.integer[Any]],
                                  patient_ids: list[str],
                                  stratify: bool = True) -> list[tuple[NDArray[np.integer[Any]], NDArray[np.integer[Any]]]]:
        """
        Criar splits de validação que garantem que ECGs do mesmo paciente
        não apareçam em treino e teste simultaneamente

        Args:
            X: Features/sinais ECG
            y: Labels/diagnósticos
            patient_ids: IDs dos pacientes para cada amostra
            stratify: Se deve manter proporção de classes

        Returns:
            Lista de tuplas (train_idx, val_idx) para cada fold
        """
        try:
            if len(X) != len(y) != len(patient_ids):
                raise ValueError("X, y e patient_ids devem ter o mesmo tamanho")

            patient_ids_array = np.array(patient_ids)
            y_array = np.array(y)

            if stratify:
                splitter = StratifiedGroupKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state
                )
                splits = list(splitter.split(X, y_array, groups=patient_ids_array))
            else:
                splitter = GroupKFold(n_splits=self.n_splits)
                splits = list(splitter.split(X, y_array, groups=patient_ids_array))

            self._validate_no_patient_leakage(splits, patient_ids_array)

            self._log_split_statistics(splits, y_array, patient_ids_array)

            return splits

        except Exception as e:
            logger.error(f"Erro ao criar splits patient-aware: {e}")
            raise

    def _validate_no_patient_leakage(self,
                                   splits: list[tuple[NDArray[np.integer[Any]], NDArray[np.integer[Any]]]],
                                   patient_ids: NDArray[np.str_]) -> None:
        """Validar que não há pacientes compartilhados entre treino e teste"""
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_patients = set(patient_ids[train_idx])
            val_patients = set(patient_ids[val_idx])

            overlap = train_patients.intersection(val_patients)
            if overlap:
                raise ValueError(
                    f"Data leakage detectado no fold {fold_idx}! "
                    f"Pacientes compartilhados: {overlap}"
                )

            logger.info(
                f"Fold {fold_idx}: {len(train_patients)} pacientes treino, "
                f"{len(val_patients)} pacientes validação"
            )

    def _log_split_statistics(self,
                            splits: list[tuple[NDArray[np.integer[Any]], NDArray[np.integer[Any]]]],
                            y: NDArray[np.integer[Any]],
                            patient_ids: NDArray[np.str_]) -> None:
        """Registrar estatísticas dos splits"""
        stats = {
            'total_samples': len(y),
            'total_patients': len(np.unique(patient_ids)),
            'class_distribution': self._get_class_distribution(y),
            'fold_stats': []
        }

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_stats = {
                'fold': fold_idx,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                'train_patients': len(np.unique(patient_ids[train_idx])),
                'val_patients': len(np.unique(patient_ids[val_idx])),
                'train_class_dist': self._get_class_distribution(y[train_idx]),
                'val_class_dist': self._get_class_distribution(y[val_idx])
            }
            if isinstance(stats['fold_stats'], list):
                stats['fold_stats'].append(fold_stats)

        self.validation_history.append(stats)
        logger.info(f"Estatísticas de validação: {stats}")

    def _get_class_distribution(self, y: NDArray[Any]) -> dict[str, int]:
        """Obter distribuição de classes"""
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique.astype(str), counts.astype(int), strict=False))

    def analyze_patient_distribution(self,
                                   patient_ids: list[str],
                                   diagnoses: list[str]) -> dict[str, Any]:
        """
        Analisar distribuição de pacientes e diagnósticos para detectar
        possíveis problemas de balanceamento

        Args:
            patient_ids: IDs dos pacientes
            diagnoses: Diagnósticos correspondentes

        Returns:
            Relatório de análise
        """
        try:
            patient_diagnoses = defaultdict(list)
            for pid, diag in zip(patient_ids, diagnoses, strict=False):
                patient_diagnoses[pid].append(diag)

            samples_per_patient = [len(diags) for diags in patient_diagnoses.values()]

            multi_diagnosis_patients = {
                pid: list(set(diags))
                for pid, diags in patient_diagnoses.items()
                if len(set(diags)) > 1
            }

            all_diagnoses = [diag for diags in patient_diagnoses.values() for diag in diags]
            diagnosis_counts = self._get_class_distribution(np.array(all_diagnoses))

            report = {
                'total_patients': len(patient_diagnoses),
                'total_samples': len(patient_ids),
                'samples_per_patient': {
                    'mean': np.mean(samples_per_patient),
                    'median': np.median(samples_per_patient),
                    'min': np.min(samples_per_patient),
                    'max': np.max(samples_per_patient),
                    'std': np.std(samples_per_patient)
                },
                'diagnosis_distribution': diagnosis_counts,
                'multi_diagnosis_patients': len(multi_diagnosis_patients),
                'multi_diagnosis_details': multi_diagnosis_patients,
                'recommendations': self._generate_validation_recommendations(
                    len(patient_diagnoses),
                    samples_per_patient,
                    diagnosis_counts
                )
            }

            return report

        except Exception as e:
            logger.error(f"Erro na análise de distribuição de pacientes: {e}")
            raise

    def _generate_validation_recommendations(self,
                                           n_patients: int,
                                           samples_per_patient: list[int],
                                           diagnosis_counts: dict[str, int]) -> list[str]:
        """Gerar recomendações para validação"""
        recommendations = []

        if n_patients < 100:
            recommendations.append(
                f"ATENÇÃO: Apenas {n_patients} pacientes únicos. "
                "Considere aumentar o dataset para validação mais robusta."
            )

        max_samples = max(samples_per_patient)
        min_samples = min(samples_per_patient)
        if max_samples / min_samples > 10:
            recommendations.append(
                "ATENÇÃO: Grande desbalanceamento no número de amostras por paciente. "
                "Considere estratégias de balanceamento."
            )

        total_samples = sum(diagnosis_counts.values())
        class_ratios = {k: v/total_samples for k, v in diagnosis_counts.items()}
        min_class_ratio = min(class_ratios.values())

        if min_class_ratio < 0.05:  # Menos de 5%
            recommendations.append(
                "ATENÇÃO: Classes muito desbalanceadas detectadas. "
                "Considere usar StratifiedGroupKFold e técnicas de balanceamento."
            )

        if n_patients < 50:
            recommendations.append(
                "RECOMENDAÇÃO: Use 3-fold CV devido ao número limitado de pacientes."
            )
        elif n_patients < 200:
            recommendations.append(
                "RECOMENDAÇÃO: Use 5-fold CV para balancear robustez e eficiência."
            )
        else:
            recommendations.append(
                "RECOMENDAÇÃO: Pode usar 10-fold CV para máxima robustez."
            )

        return recommendations

    def create_temporal_splits(self,
                             timestamps: list[str],
                             patient_ids: list[str],
                             test_ratio: float = 0.2) -> tuple[NDArray[np.integer[Any]], NDArray[np.integer[Any]]]:
        """
        Criar split temporal para simular cenário de produção
        (treinar em dados antigos, testar em dados recentes)

        Args:
            timestamps: Timestamps dos ECGs
            patient_ids: IDs dos pacientes
            test_ratio: Proporção para teste

        Returns:
            Tupla (train_idx, test_idx)
        """
        try:
            timestamps_array = np.array(timestamps)
            patient_ids_array = np.array(patient_ids)

            sorted_indices = np.argsort(timestamps_array)

            n_total = len(timestamps_array)
            n_test = int(n_total * test_ratio)

            test_candidates = sorted_indices[-n_test:]
            test_patients = set(patient_ids_array[test_candidates])

            train_mask = ~np.isin(patient_ids_array, list(test_patients))
            train_idx = np.where(train_mask)[0]
            test_idx = test_candidates

            train_patients = set(patient_ids_array[train_idx])
            overlap = train_patients.intersection(test_patients)

            if overlap:
                logger.warning(f"Overlap detectado no split temporal: {overlap}")

            logger.info(
                f"Split temporal: {len(train_idx)} treino, {len(test_idx)} teste, "
                f"{len(train_patients)} pacientes treino, {len(test_patients)} pacientes teste"
            )

            return train_idx, test_idx

        except Exception as e:
            logger.error(f"Erro ao criar split temporal: {e}")
            raise

    def get_validation_report(self) -> dict[str, Any]:
        """Obter relatório completo das validações realizadas"""
        if not self.validation_history:
            return {"message": "Nenhuma validação realizada ainda"}

        latest_validation = self.validation_history[-1]

        fold_train_sizes = [f['train_samples'] for f in latest_validation['fold_stats']]
        fold_val_sizes = [f['val_samples'] for f in latest_validation['fold_stats']]

        report = {
            'total_validations': len(self.validation_history),
            'latest_validation': latest_validation,
            'fold_size_consistency': {
                'train_std': np.std(fold_train_sizes),
                'val_std': np.std(fold_val_sizes),
                'train_cv': np.std(fold_train_sizes) / np.mean(fold_train_sizes),
                'val_cv': np.std(fold_val_sizes) / np.mean(fold_val_sizes)
            },
            'recommendations': []
        }

        recommendations = report.get('recommendations')
        fold_consistency = report.get('fold_size_consistency')
        if (isinstance(recommendations, list) and
            isinstance(fold_consistency, dict) and
            fold_consistency.get('train_cv', 0) > 0.1):
            recommendations.append(
                "ATENÇÃO: Grande variação no tamanho dos folds de treino. "
                "Verifique a distribuição de pacientes."
            )

        return report
