import type { ExperimentAnalysis } from '../types'

export interface PerformanceMetricItem {
  id: string
  labelKey: string
  value?: number
}

export interface PerformanceSelection {
  labelKey: string
  value?: number
  bestEpoch?: number
}

export interface PerformanceSummaryModel {
  isMultilabel: boolean
  selection: PerformanceSelection
  primary: PerformanceMetricItem[]
  testSupplemental: PerformanceMetricItem[]
  validationSupplemental: PerformanceMetricItem[]
}

function finiteNumber(value: unknown): number | undefined {
  if (value === null || value === undefined || value === '') return undefined
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue : undefined
}

function firstFinite(...values: unknown[]): number | undefined {
  for (const value of values) {
    const numberValue = finiteNumber(value)
    if (numberValue !== undefined) return numberValue
  }
  return undefined
}

function splitMetrics(analysis: ExperimentAnalysis | undefined, split: 'validation' | 'test') {
  return analysis?.metrics?.[split] ?? {}
}

function curveMetric(
  analysis: ExperimentAnalysis | undefined,
  split: 'validation' | 'test',
  metric: 'auc' | 'aupr',
) {
  const curves = analysis?.curves?.[split]
  return metric === 'auc' ? curves?.roc?.auc : curves?.pr?.aupr
}

export function isMultilabelEvaluation(taskType: string, analysis?: ExperimentAnalysis): boolean {
  const point = analysis?.metrics?.operating_point
  const thresholds = point?.thresholds ?? point?.values ?? []
  const validationAverage = String(splitMetrics(analysis, 'validation').averaging ?? '')
  return thresholds.length > 1 || validationAverage === 'macro' || taskType === 'pep_function'
}

function item(id: string, labelKey: string, value: unknown): PerformanceMetricItem {
  return { id, labelKey, value: finiteNumber(value) }
}

function available(items: PerformanceMetricItem[]) {
  return items.filter((entry) => entry.value !== undefined)
}

export function buildPerformanceSummary(
  summary: Record<string, unknown>,
  analysis: ExperimentAnalysis | undefined,
  taskType: string,
): PerformanceSummaryModel {
  const isMultilabel = isMultilabelEvaluation(taskType, analysis)
  const validation = splitMetrics(analysis, 'validation')
  const test = splitMetrics(analysis, 'test')
  const bestEpoch = finiteNumber(summary.best_epoch)

  if (isMultilabel) {
    return {
      isMultilabel,
      selection: {
        labelKey: 'detail.bestValMacroF1',
        value: firstFinite(summary.best_val_f1, summary.val_f1, validation.f1),
        bestEpoch,
      },
      primary: [
        item('macro_f1', 'detail.testMacroF1', firstFinite(summary.test_f1, test.f1)),
        item('sacc', 'detail.testSacc', firstFinite(summary.test_sacc, test.sacc)),
        item('macro_auc', 'detail.testMacroAuc', firstFinite(summary.test_auc, test.auc)),
        item('macro_aupr', 'detail.testMacroAupr', firstFinite(summary.test_aupr, test.aupr)),
      ],
      testSupplemental: available([
        item('acc', 'detail.testAcc', firstFinite(summary.test_acc, test.acc)),
        item(
          'balanced_accuracy',
          'detail.testBalancedAccuracy',
          firstFinite(summary.test_balanced_accuracy, test.balanced_accuracy),
        ),
        item(
          'precision',
          'detail.testPrecision',
          firstFinite(summary.test_precision, test.precision),
        ),
        item('recall', 'detail.testRecall', firstFinite(summary.test_recall, test.recall)),
        item(
          'specificity',
          'detail.testSpecificity',
          firstFinite(summary.test_specificity, test.specificity),
        ),
        item('mcc', 'detail.testMcc', firstFinite(summary.test_mcc, test.mcc)),
        item(
          'micro_auc',
          'detail.testMicroAuc',
          firstFinite(summary.test_auc_micro, test.auc_micro),
        ),
        item(
          'micro_aupr',
          'detail.testMicroAupr',
          firstFinite(summary.test_aupr_micro, test.aupr_micro),
        ),
      ]),
      validationSupplemental: available([
        item('acc', 'detail.validationAcc', firstFinite(summary.val_acc, validation.acc)),
        item('sacc', 'detail.validationSacc', firstFinite(summary.val_sacc, validation.sacc)),
        item(
          'macro_auc',
          'detail.validationMacroAuc',
          firstFinite(summary.val_auc, validation.auc),
        ),
        item(
          'macro_aupr',
          'detail.validationMacroAupr',
          firstFinite(summary.val_aupr, validation.aupr),
        ),
        item(
          'balanced_accuracy',
          'detail.validationBalancedAccuracy',
          firstFinite(summary.val_balanced_accuracy, validation.balanced_accuracy),
        ),
        item(
          'precision',
          'detail.validationPrecision',
          firstFinite(summary.val_precision, validation.precision),
        ),
        item(
          'recall',
          'detail.validationRecall',
          firstFinite(summary.val_recall, validation.recall),
        ),
        item(
          'specificity',
          'detail.validationSpecificity',
          firstFinite(summary.val_specificity, validation.specificity),
        ),
        item('mcc', 'detail.validationMcc', firstFinite(summary.val_mcc, validation.mcc)),
        item(
          'micro_auc',
          'detail.validationMicroAuc',
          firstFinite(summary.val_auc_micro, validation.auc_micro),
        ),
        item(
          'micro_aupr',
          'detail.validationMicroAupr',
          firstFinite(summary.val_aupr_micro, validation.aupr_micro),
        ),
      ]),
    }
  }

  return {
    isMultilabel,
    selection: {
      labelKey: 'detail.bestValAuc',
      value: firstFinite(
        summary.best_val_auc,
        summary.val_auc,
        validation.auc,
        curveMetric(analysis, 'validation', 'auc'),
      ),
      bestEpoch,
    },
    primary: [
      item(
        'auc',
        'detail.testAuc',
        firstFinite(summary.test_auc, test.auc, curveMetric(analysis, 'test', 'auc')),
      ),
      item(
        'aupr',
        'detail.testAupr',
        firstFinite(summary.test_aupr, test.aupr, curveMetric(analysis, 'test', 'aupr')),
      ),
      item('f1', 'detail.testF1', firstFinite(summary.test_f1, test.f1)),
      item('acc', 'detail.testAcc', firstFinite(summary.test_acc, test.acc)),
    ],
    testSupplemental: available([
      item(
        'balanced_accuracy',
        'detail.testBalancedAccuracy',
        firstFinite(summary.test_balanced_accuracy, test.balanced_accuracy),
      ),
      item(
        'precision',
        'detail.testPrecision',
        firstFinite(summary.test_precision, test.precision),
      ),
      item('recall', 'detail.testRecall', firstFinite(summary.test_recall, test.recall)),
      item(
        'specificity',
        'detail.testSpecificity',
        firstFinite(summary.test_specificity, test.specificity),
      ),
      item('mcc', 'detail.testMcc', firstFinite(summary.test_mcc, test.mcc)),
    ]),
    validationSupplemental: available([
      item('acc', 'detail.validationAcc', firstFinite(summary.val_acc, validation.acc)),
      item('f1', 'detail.validationF1', firstFinite(summary.val_f1, validation.f1)),
      item(
        'aupr',
        'detail.validationAupr',
        firstFinite(summary.val_aupr, validation.aupr, curveMetric(analysis, 'validation', 'aupr')),
      ),
      item(
        'balanced_accuracy',
        'detail.validationBalancedAccuracy',
        firstFinite(summary.val_balanced_accuracy, validation.balanced_accuracy),
      ),
      item(
        'precision',
        'detail.validationPrecision',
        firstFinite(summary.val_precision, validation.precision),
      ),
      item('recall', 'detail.validationRecall', firstFinite(summary.val_recall, validation.recall)),
      item(
        'specificity',
        'detail.validationSpecificity',
        firstFinite(summary.val_specificity, validation.specificity),
      ),
      item('mcc', 'detail.validationMcc', firstFinite(summary.val_mcc, validation.mcc)),
    ]),
  }
}
