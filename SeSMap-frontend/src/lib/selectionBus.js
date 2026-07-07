// src/lib/selectionBus.js

// —— 实时选中（预览）总线 —— //
const subsLive = new Set();
/**
 * 订阅“实时选中集”变化（例如 hover/点击后右侧即时预览）
 * @param {(payload:{nodes:any[], links:any[]})=>void} fn
 * @returns {()=>void}
 */
export function onSelectionChange(fn) {
  subsLive.add(fn);
  return () => subsLive.delete(fn);
}
/** 推送实时选中集 */
export function emitSelection(payload) {
  const p = payload || { nodes: [], links: [] };
  subsLive.forEach(fn => { try { fn(p); } catch(e){ console.warn(e); } });
}

// —— Save 后“步骤快照”总线 —— //
const subsSaved = new Set();
/**
 * 订阅“保存步骤”的事件（点 Save 后堆叠到右侧 Steps）
 * @param {(payload:{
 *   title?: string, createdAt?: number,
 *   nodes:any[], links:any[],
 *   rawText?: string, summary?: string, meta?: any
 * })=>void} fn
 * @returns {()=>void}
 */
export function onSelectionSaved(fn) {
  subsSaved.add(fn);
  return () => subsSaved.delete(fn);
}
/** 推送保存的步骤快照 */
export function emitSelectionSaved(payload) {
  const p = payload || { nodes: [], links: [] };
  subsSaved.forEach(fn => { try { fn(p); } catch(e){ console.warn(e); } });
}

// ===== summarize-selected bus =====
const _summarizeHandlers = new Set();

/** 右侧点击“Summarize Selected”时触发 */
export function emitSummarizeSelected(payload) {
  // payload: { stepId, title, nodes: [{key, panelIdx,q,r, msus:[{id,text,checked}]}], selectedTexts: string[] }
  _summarizeHandlers.forEach(fn => fn && fn(payload));
}

/** 上游（左侧或服务层）订阅后做实际 LLM 调用 */
export function onSummarizeSelected(handler) {
  _summarizeHandlers.add(handler);
  return () => _summarizeHandlers.delete(handler);
}

// ===== stepwise MSU semantic filter bus =====
const _stepwiseMsuCandidateProviders = new Set();
const _stepwiseMsuFilterHandlers = new Set();

/**
 * 右侧 LinkCard 注册当前可筛选的 MSU 候选。
 * @param {()=>Array<any>} provider
 * @returns {()=>void}
 */
export function onStepwiseMsuCandidates(provider) {
  _stepwiseMsuCandidateProviders.add(provider);
  return () => _stepwiseMsuCandidateProviders.delete(provider);
}

/** 收集所有已挂载 Stepwise 卡片中的 MSU 候选 */
export function collectStepwiseMsuCandidates() {
  const out = [];
  _stepwiseMsuCandidateProviders.forEach(provider => {
    try {
      const items = provider?.();
      if (Array.isArray(items)) out.push(...items);
    } catch (e) {
      console.warn('[selectionBus] collectStepwiseMsuCandidates failed:', e);
    }
  });
  return out;
}

/**
 * 右侧 LinkCard 注册“按 uid 勾选 MSU”的处理器。
 * @param {(payload:{uids?:string[], matchedUids?:string[], query?:string, intent?:string})=>any} handler
 * @returns {()=>void}
 */
export function onApplyStepwiseMsuFilter(handler) {
  _stepwiseMsuFilterHandlers.add(handler);
  return () => _stepwiseMsuFilterHandlers.delete(handler);
}

/** 将语义筛选结果广播给所有 Stepwise 卡片，只新增勾选，不取消已有选择 */
export function emitApplyStepwiseMsuFilter(payload = {}) {
  const result = {
    cards: 0,
    matched: 0,
    newlyChecked: 0,
    alreadyChecked: 0
  };
  _stepwiseMsuFilterHandlers.forEach(handler => {
    try {
      const r = handler?.(payload) || {};
      result.cards += 1;
      result.matched += Number(r.matched) || 0;
      result.newlyChecked += Number(r.newlyChecked) || 0;
      result.alreadyChecked += Number(r.alreadyChecked) || 0;
    } catch (e) {
      console.warn('[selectionBus] emitApplyStepwiseMsuFilter failed:', e);
    }
  });
  return result;
}
