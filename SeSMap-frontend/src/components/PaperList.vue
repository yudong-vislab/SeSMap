<script setup>
import { ref, watch, nextTick, computed, defineExpose, onMounted, onBeforeUnmount } from 'vue'

/* ---------- Props ---------- */
const props = defineProps({
  /** 选项卡标题（双击可编辑） */
  title: { type: String, default: 'Paper Query' },
  /** 论文项：[{ id, globalIndex, name, year, count, content(imgUrl), pdfUrl }] */
  items: { type: Array, default: () => [] },
  /** 多分组：[{ key?, title, items:[同上结构] }] —— 传入则按分组渲染，覆盖单 items 模式 */
  groups: { type: Array, default: () => [] },
  /** items 为空时是否启用内置 demo（默认 false） */
  useDemo: { type: Boolean, default: false },
  /** 有选中时，其它项透明度 */
  dimOpacity: { type: Number, default: 0.15 },
  /** v-model:selected-ids（用 globalIndex 做主键） */
  selectedIds: { type: Array, default: () => [] },
  /** 每个缩略卡最小宽度（控制每行个数），建议 92~120px */
  tileMin: { type: [String, Number], default: 100 },
  /** 缩略图高宽比（h = tileMin * thumbRatio） */
  thumbRatio: { type: Number, default: 0.68 },
  /** Semantic map color snapshot: { colorByCountry, colorByPanelCountry, normalizeCountryId } */
  colorMaps: { type: Object, default: () => ({}) }
})

/* ---------- Emits ---------- */
const emit = defineEmits(['update:title', 'update:selectedIds', 'open-pdf', 'close', 'update:groups'])

/* ---------- 演示数据（兜底） ---------- */
const demoItems = [
  {
    id: 0, globalIndex: 0, name: 'Compass', year: '2019', count: 38,
    content: new URL('../assets/pictures/air/Compass Towards Better Causal Analysis of Urban Time Series.png', import.meta.url).href,
    pdfUrl: new URL('../assets/pdf/case2/Compass.pdf', import.meta.url).href
  },
  {
    id: 1, globalIndex: 1, name: 'GeoChron', year: '2024', count: 52,
    content: new URL('../assets/pictures/air/Visualizing Large-Scale Spatial Time Series with GeoChron.png', import.meta.url).href,
    pdfUrl: new URL('../assets/pdf/case2/GeoChron.pdf', import.meta.url).href
  },
  {
    id: 2, globalIndex: 2, name: 'VolumeSTCube', year: '2020', count: 45,
    content: new URL('../assets/pictures/air/Volume-Based Space-Time Cube for Large-Scale Continuous Spatial Time Series.png', import.meta.url).href,
    pdfUrl: new URL('../assets/pdf/case2/VolumeSTCube-TVCG.pdf', import.meta.url).href
  },
  {
    id: 3, globalIndex: 3, name: 'Absorbing aerosols', year: '2024', count: 41,
    content: new URL('../assets/pictures/air/Threefold reduction of modeled uncertainty in direct radiative effects over biomass burning regions by constraining absorbing aerosols.png', import.meta.url).href,
    pdfUrl: new URL('../assets/pdf/case2/sciadv.adi3568.pdf', import.meta.url).href
  },
  {
    id: 4, globalIndex: 4, name: 'WRF-Chem PM2.5', year: '2024', count: 35,
    content: new URL('../assets/pictures/air/Improving WRF-Chem PM2.5 predictions by combining data assimilation and deep-learning-based bias correction.png', import.meta.url).href,
    pdfUrl: new URL('../assets/pdf/case2/1-s2.0-S0160412024007864-main.pdf', import.meta.url).href
  }
]

/* ---------- 计算 ---------- */
const hasGroups = computed(() => Array.isArray(props.groups) && props.groups.length > 0)
const displayItems = computed(() =>
  (props.items && props.items.length) ? props.items : (props.useDemo ? demoItems : [])
)

/* ---------- 标题编辑（单列表） ---------- */
const editing = ref(false)
const titleLocal = ref(props.title)
watch(() => props.title, v => (titleLocal.value = v))

function startEdit() {
  editing.value = true
  nextTick(() => {
    const el = document.querySelector('.paperlist-card.single .step__title-text')
    if (!el) return
    const range = document.createRange()
    range.selectNodeContents(el)
    range.collapse(false)
    const sel = window.getSelection()
    sel.removeAllRanges()
    sel.addRange(range)
    el.focus()
  })
}

function finishEdit(e) {
  if (!editing.value) return
  const t = (e?.target?.innerText || '').trim()
  if (t && t !== props.title) emit('update:title', t)
  else if (e?.target) e.target.innerText = props.title
  editing.value = false
}

function onTitleKey(e) {
  if (!editing.value) return
  if (e.key === 'Enter') { e.preventDefault(); e.currentTarget.blur() }
  if (e.key === 'Escape') {
    e.preventDefault()
    e.currentTarget.innerText = props.title
    editing.value = false
    e.currentTarget.blur()
  }
}

/* ---------- 分组标题编辑（新增：每组可编辑） ---------- */
/* 本地镜像，避免直接改 props */
const groupsLocal = ref([])
watch(
  () => props.groups,
  (g) => { groupsLocal.value = Array.isArray(g) ? g.map(x => ({ ...x })) : [] },
  { immediate: true, deep: true }
)

const editingPerGroup = ref([]) // boolean[]
function ensureGp(idx) { if (!Array.isArray(editingPerGroup.value)) editingPerGroup.value = []; if (editingPerGroup.value[idx] == null) editingPerGroup.value[idx] = false }

function startEditGroup(gi) {
  ensureGp(gi)
  editingPerGroup.value[gi] = true
  nextTick(() => {
    const el = document.querySelector(`.paperlist-card.group[data-gi="${gi}"] .step__title-text`)
    if (!el) return
    const range = document.createRange()
    range.selectNodeContents(el)
    range.collapse(false)
    const sel = window.getSelection()
    sel.removeAllRanges()
    sel.addRange(range)
    el.focus()
  })
}

function finishEditGroup(e, gi) {
  ensureGp(gi)
  if (!editingPerGroup.value[gi]) return
  const t = (e?.target?.innerText || '').trim()
  const fallback = `${titleLocal.value} · ${gi + 1}`
  const newTitle = t || fallback
  // 本地更新
  if (!groupsLocal.value[gi]) groupsLocal.value[gi] = {}
  groupsLocal.value[gi].title = newTitle
  // 通知父层（可选接收）
  emit('update:groups', groupsLocal.value)
  // 若清空则回退显示
  if (e?.target) e.target.innerText = newTitle
  editingPerGroup.value[gi] = false
}

function onTitleKeyGroup(e, gi) {
  if (!editingPerGroup.value[gi]) return
  if (e.key === 'Enter') { e.preventDefault(); e.currentTarget.blur() }
  if (e.key === 'Escape') {
    e.preventDefault()
    const fallback = `${titleLocal.value} · ${gi + 1}`
    const old = groupsLocal.value[gi]?.title || fallback
    e.currentTarget.innerText = old
    editingPerGroup.value[gi] = false
    e.currentTarget.blur()
  }
}

function closeGroup(gi) {
  const next = groupsLocal.value.filter((_, idx) => idx !== gi)
  groupsLocal.value = next
  editingPerGroup.value = editingPerGroup.value.filter((_, idx) => idx !== gi)
  selectedByGroup.value = selectedByGroup.value.filter((_, idx) => idx !== gi)
  emit('update:groups', next)
}

/* ---------- 选择（单/分组） ---------- */
// 单列表
const selected = ref([...(props.selectedIds || [])])
watch(() => props.selectedIds, v => (selected.value = Array.isArray(v) ? [...v] : []))
function toggleSelect(gi) {
  const s = new Set(selected.value)
  s.has(gi) ? s.delete(gi) : s.add(gi)
  selected.value = [...s]
  emit('update:selectedIds', selected.value)
}
function clearSelection() { selected.value = []; emit('update:selectedIds', selected.value) }
function opacityFor(gi) { return !selected.value.length ? 1 : (selected.value.includes(gi) ? 1 : props.dimOpacity) }

// 分组
const selectedByGroup = ref([]) // [[gi,...], ...]
watch(hasGroups, (on) => { if (on && !selectedByGroup.value.length) selectedByGroup.value = props.groups.map(() => []) })
watch(
  groupsLocal,
  (g) => {
    selectedByGroup.value = g.map((_, idx) => selectedByGroup.value[idx] || [])
    editingPerGroup.value = g.map((_, idx) => editingPerGroup.value[idx] || false)
  },
  { deep: true }
)
function toggleSelectInGroup(groupIdx, gi) {
  if (!selectedByGroup.value[groupIdx]) selectedByGroup.value[groupIdx] = []
  const s = new Set(selectedByGroup.value[groupIdx])
  s.has(gi) ? s.delete(gi) : s.add(gi)
  selectedByGroup.value[groupIdx] = [...s]
}
function opacityForGroup(groupIdx, gi) {
  const arr = selectedByGroup.value[groupIdx] || []
  return !arr.length ? 1 : (arr.includes(gi) ? 1 : props.dimOpacity)
}

function openPdf(it) { emit('open-pdf', it) }

const DEFAULT_PAPER_DOT = '#DCDCDC'
const GALLERY_COLOR_PRESETS = ['#4C78A8', '#72B7B2', '#54A24B', '#B279A2', '#E45756', '#F58518', '#EECA3B']
function pickMapValue(mapLike, key) {
  if (!mapLike || key == null) return null
  const keys = [key, String(key)]
  if (mapLike instanceof Map) {
    for (const k of keys) if (mapLike.has(k)) return mapLike.get(k)
    return null
  }
  if (typeof mapLike === 'object') {
    for (const k of keys) if (Object.prototype.hasOwnProperty.call(mapLike, k)) return mapLike[k]
  }
  return null
}
function countryIdOfItem(it) {
  return it?.semanticCountryId ?? it?.meta?.semanticCountryId ?? it?.countryId ?? it?.paperCountryId ?? it?.meta?.countryId ?? it?.meta?.paperCountryId ?? it?.paper_id ?? it?.paperId ?? null
}
function projectIdOfItem(it) {
  return it?.projectId ?? it?.meta?.projectId ?? it?.project_id ?? it?.meta?.project_id ?? null
}
function sourceKeyOfItem(it) {
  return it?.sourceKey ?? it?.meta?.sourceKey ?? null
}
function paperColorFor(it) {
  const raw = countryIdOfItem(it)
  if (raw == null || raw === '') return DEFAULT_PAPER_DOT
  const normalize = props.colorMaps?.normalizeCountryId
  const cid = typeof normalize === 'function' ? normalize(raw) : raw
  const projectId = projectIdOfItem(it)
  const sourceKey = sourceKeyOfItem(it) || (projectId ? `${projectId}|${cid}` : null)
  return pickMapValue(props.colorMaps?.sourceColorByKey, sourceKey) || DEFAULT_PAPER_DOT
}

const galleryColorMenu = ref({
  open: false,
  x: 0,
  y: 0,
  item: null,
  color: GALLERY_COLOR_PRESETS[0],
  hex: GALLERY_COLOR_PRESETS[0]
})

const galleryColorMenuStyle = computed(() => ({
  left: `${galleryColorMenu.value.x}px`,
  top: `${galleryColorMenu.value.y}px`
}))

function normalizeHexColor(color, fallback = GALLERY_COLOR_PRESETS[0]) {
  const s = String(color || '').trim()
  if (/^#[0-9a-f]{6}$/i.test(s)) return s.toUpperCase()
  if (/^#[0-9a-f]{3}$/i.test(s)) {
    const h = s.slice(1)
    return `#${h[0]}${h[0]}${h[1]}${h[1]}${h[2]}${h[2]}`.toUpperCase()
  }
  return fallback
}

function normalizedCountryIdForItem(it) {
  const raw = countryIdOfItem(it)
  if (raw == null || raw === '') return null
  const normalize = props.colorMaps?.normalizeCountryId
  return typeof normalize === 'function' ? normalize(raw) : raw
}

function openGalleryColorMenu(e, it) {
  const cid = normalizedCountryIdForItem(it)
  if (cid == null || cid === '') return
  const current = paperColorFor(it)
  const width = 180
  const height = 118
  const pad = 10
  const color = normalizeHexColor(current, GALLERY_COLOR_PRESETS[0])
  const x = Math.max(pad, Math.min(e.clientX, window.innerWidth - width - pad))
  const y = Math.max(pad, Math.min(e.clientY, window.innerHeight - height - pad))
  galleryColorMenu.value = {
    open: true,
    x,
    y,
    item: it,
    color,
    hex: color
  }
}

function closeGalleryColorMenu() {
  galleryColorMenu.value.open = false
}

function setGalleryMenuColor(color) {
  const next = normalizeHexColor(color, galleryColorMenu.value.color)
  galleryColorMenu.value.color = next
  galleryColorMenu.value.hex = next
}

function setGalleryMenuHex(value) {
  const next = String(value || '').trim().toUpperCase()
  galleryColorMenu.value.hex = next
  if (/^#[0-9A-F]{6}$/.test(next)) {
    galleryColorMenu.value.color = next
  }
}

function randomGalleryColor() {
  const color = GALLERY_COLOR_PRESETS[Math.floor(Math.random() * GALLERY_COLOR_PRESETS.length)]
  setGalleryMenuColor(color)
}

function applyGalleryColor() {
  const item = galleryColorMenu.value.item
  const cid = normalizedCountryIdForItem(item)
  const color = normalizeHexColor(galleryColorMenu.value.hex, galleryColorMenu.value.color)
  if (cid == null || cid === '') return closeGalleryColorMenu()

  const ctrl = window?.SemanticMapCtrl || window?.SemanticMap || null
  const applied = ctrl?.setSourceColor?.(cid, color)
  if (!applied) {
    window?.dispatchEvent?.(new CustomEvent('semanticmap:sourcecolorrequest', {
      detail: { countryId: cid, color }
    }))
  }
  closeGalleryColorMenu()
}

function onGlobalPointerDown(e) {
  if (!galleryColorMenu.value.open) return
  if (e?.target?.closest?.('.paper-color-menu')) return
  closeGalleryColorMenu()
}

function onGlobalEscape(e) {
  if (e.key === 'Escape') closeGalleryColorMenu()
}

onMounted(() => {
  window.addEventListener('pointerdown', onGlobalPointerDown, true)
  window.addEventListener('keydown', onGlobalEscape)
  window.addEventListener('scroll', closeGalleryColorMenu, true)
})

onBeforeUnmount(() => {
  window.removeEventListener('pointerdown', onGlobalPointerDown, true)
  window.removeEventListener('keydown', onGlobalEscape)
  window.removeEventListener('scroll', closeGalleryColorMenu, true)
})

/* ---------- CSS 变量 ---------- */
const gridVars = computed(() => {
  const min = typeof props.tileMin === 'number' ? `${props.tileMin}px` : String(props.tileMin)
  const ratio = String(props.thumbRatio)
  return { '--tile-min': min, '--thumb-ratio': ratio }
})

defineExpose({ clearSelection })
</script>

<template>
  <!-- ============ 分组模式：每个分组独立卡片 + 可编辑标题 ============ -->
  <template v-if="hasGroups">
    <article
      v-for="(g, gi) in groupsLocal"
      :key="g.key ?? gi"
      class="step-card paperlist-card group"
      :data-gi="gi"
    >
      <!-- 标题栏（可编辑） -->
      <div class="step__title-row">
        <div
          class="step__title step__title-text"
          :contenteditable="editingPerGroup[gi] ? 'plaintext-only' : 'false'"
          @dblclick="startEditGroup(gi)"
          @blur="(e) => finishEditGroup(e, gi)"
          @keydown="(e) => onTitleKeyGroup(e, gi)"
          :title="editingPerGroup[gi] ? 'Enter to save · Esc to cancel' : 'Double-click to edit'"
        >
          {{ g.title || (titleLocal + ' · ' + (gi + 1)) }}
        </div>
        <button
          class="paperlist-close"
          type="button"
          title="Remove this paper group"
          aria-label="Remove this paper group"
          @click.stop="closeGroup(gi)"
        >
          ×
        </button>
      </div>

      <section class="subcards__item subcard subcard_paperlist">
        <div class="pl-grid" :style="gridVars">
          <div
            v-for="it in (g.items || [])"
            :key="it.globalIndex"
            class="pl-item"
            :class="{ 'is-selected': (selectedByGroup[gi] || []).includes(it.globalIndex) }"
            :style="{ opacity: opacityForGroup(gi, it.globalIndex) }"
            @click.stop="toggleSelectInGroup(gi, it.globalIndex)"
            @contextmenu.prevent.stop="openGalleryColorMenu($event, it)"
            :title="it.name"
          >
            <div class="thumb" :style="{ backgroundImage: `url(${it.content || it.thumbUrl || ''})` }">
              <button class="eye-btn" @click.stop="openPdf(it)" title="Preview PDF" aria-label="Preview">
                <svg viewBox="0 0 24 24" class="eye-svg" aria-hidden="true">
                  <path d="M12 4.5c-6.627 0-12 7.072-12 7.5s5.373 7.5 12 7.5 12-7.072 12-7.5-5.373-7.5-12-7.5zm0 12c-2.485 0-4.5-2.015-4.5-4.5s2.015-4.5 4.5-4.5 4.5 2.015 4.5 4.5-2.015 4.5-4.5 4.5zm0-7.5c-1.657 0-3 1.343-3 3s1.343 3 3 3 3-1.343 3-3-1.343-3-3-3z"/>
                </svg>
              </button>
            </div>
            <div class="meta">
              <span
                class="paper-dot"
                :style="{ backgroundColor: paperColorFor(it) }"
                :title="it.name || it.title"
              ></span>
              <span class="title">{{ it.name || it.title }}</span>
            </div>
          </div>
        </div>
      </section>
    </article>
  </template>

  <!-- ============ 单列表模式：可编辑标题 + 网格 ============ -->
  <article v-else class="step-card paperlist-card single">
    <div class="step__title-row">
      <div
        class="step__title step__title-text"
        :contenteditable="editing ? 'plaintext-only' : 'false'"
        @dblclick="startEdit"
        @blur="finishEdit"
        @keydown="onTitleKey"
        :title="editing ? 'Enter to save · Esc to cancel' : 'Double-click to edit'"
      >
        {{ titleLocal }}
      </div>
      <!-- <button class="subspace-close" @click="$emit('close')" aria-label="Close">×</button> -->
    </div>

    <section class="subcards__item subcard subcard_paperlist">
      <div class="pl-grid" :style="gridVars">
        <div
          v-for="it in displayItems"
          :key="it.globalIndex"
          class="pl-item"
          :class="{ 'is-selected': selected.includes(it.globalIndex) }"
          :style="{ opacity: opacityFor(it.globalIndex) }"
          @click.stop="toggleSelect(it.globalIndex)"
          @contextmenu.prevent.stop="openGalleryColorMenu($event, it)"
          :title="it.name"
        >
          <div class="thumb" :style="{ backgroundImage: `url(${it.content || it.thumbUrl || ''})` }">
            <button class="eye-btn" @click.stop="openPdf(it)" title="Preview PDF" aria-label="Preview">
              <svg viewBox="0 0 24 24" class="eye-svg" aria-hidden="true">
                <path d="M12 4.5c-6.627 0-12 7.072-12 7.5s5.373 7.5 12 7.5 12-7.072 12-7.5-5.373-7.5-12-7.5zm0 12c-2.485 0-4.5-2.015-4.5-4.5s2.015-4.5 4.5-4.5 4.5 2.015 4.5 4.5-2.015 4.5-4.5 4.5zm0-7.5c-1.657 0-3 1.343-3 3s1.343 3 3 3 3-1.343 3-3-1.343-3-3-3z"/>
            </svg>
            </button>
          </div>
          <div class="meta">
            <span
              class="paper-dot"
              :style="{ backgroundColor: paperColorFor(it) }"
              :title="it.name || it.title"
            ></span>
            <span class="title">{{ it.name || it.title }}</span>
          </div>
        </div>
      </div>
    </section>
  </article>

  <div
    v-if="galleryColorMenu.open"
    class="paper-color-menu"
    :style="galleryColorMenuStyle"
    @click.stop
    @contextmenu.prevent.stop
  >
    <div class="paper-color-menu__row">
      <input
        class="paper-color-menu__input"
        type="color"
        :value="galleryColorMenu.color"
        aria-label="Source color"
        @input="setGalleryMenuColor($event.target.value)"
      />
      <input
        class="paper-color-menu__hex"
        type="text"
        spellcheck="false"
        :value="galleryColorMenu.hex"
        aria-label="Hex color"
        @input="setGalleryMenuHex($event.target.value)"
      />
      <button class="paper-color-menu__random" type="button" title="Random color" aria-label="Random color" @click="randomGalleryColor">R</button>
    </div>
    <div class="paper-color-menu__swatches">
      <button
        v-for="color in GALLERY_COLOR_PRESETS"
        :key="color"
        class="paper-color-menu__swatch"
        type="button"
        :style="{ backgroundColor: color }"
        :aria-label="`Use ${color}`"
        @click="setGalleryMenuColor(color)"
      ></button>
    </div>
    <div class="paper-color-menu__actions">
      <button class="paper-color-menu__button" type="button" @click="closeGalleryColorMenu">Cancel</button>
      <button class="paper-color-menu__button paper-color-menu__button_primary" type="button" @click="applyGalleryColor">Apply</button>
    </div>
  </div>
</template>

<style scoped>
/* 卡片基础 */
.paperlist-card{ overflow: hidden; }
.step-card{
  border:1px solid #e5e7eb;
  border-radius:10px;
  padding:6px;
  margin-bottom:10px;
  display:grid;
  gap:6px;
  grid-template-rows:auto auto;
  background:#fff;
  font-family:var(--app-font);
}

/* 标题行（单列表/分组都显示，可编辑） */
.step__title-row{ display:flex; align-items:center; gap:6px; min-width:0; }
.step__title{
  flex:1 1 auto;
  min-width:0;
  font-weight:650; font-size:11px; line-height:1.25;
  padding:4px 6px; border-radius:8px;
  background:#f9fafb;
  user-select:text; cursor:text;
  outline:none; border:1px dashed transparent;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.step__title[contenteditable="plaintext-only"]{
  border-color:#c7d2fe; background:#eef2ff;
  white-space:normal;
  overflow:visible;
}
.paperlist-close{
  flex:0 0 18px;
  position:relative;
  width:18px;
  height:18px;
  min-width:18px;
  max-width:18px;
  min-height:18px;
  max-height:18px;
  padding:0;
  margin:0;
  appearance:none;
  -webkit-appearance:none;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  border:1px solid #ddd;
  border-radius:999px;
  background:#fff;
  color:#333;
  cursor:pointer;
  font-size:0;
  line-height:1;
  box-sizing:border-box;
}
.paperlist-close::before,
.paperlist-close::after{
  content:'';
  position:absolute;
  left:50%;
  top:50%;
  width:8px;
  height:1.45px;
  border-radius:999px;
  background:currentColor;
  transform-origin:center;
}
.paperlist-close::before{ transform:translate(-50%, -50%) rotate(45deg); }
.paperlist-close::after{ transform:translate(-50%, -50%) rotate(-45deg); }
.paperlist-close:hover{
  background:darkred;
  color:#fff;
}

/* 内容卡壳 */
.subcards__item{ border-radius:10px; }
.subcard_paperlist{
  border:1px solid #e5e7eb; background:#fff;
  border-radius:10px; padding:8px;
}

/* 网格 */
.pl-grid{
  --tile-min: 100px;
  --thumb-ratio: 0.68; /* 高度 = tileMin * ratio */
  display:grid;
  grid-template-columns: repeat(auto-fill, minmax(var(--tile-min), 1fr));
  gap:4px;
  align-items:start;
}

/* 单卡片 */
.pl-item{
  user-select:none; background:#fff;
  border:1px solid #e5e7eb; border-radius:8px;
  padding:6px; transition:border-color .12s ease, box-shadow .12s ease, transform .06s ease;
}
.pl-item:hover{ border-color:#d1d5db; box-shadow:0 2px 8px rgba(0,0,0,.05); transform: translateY(-1px); }
.pl-item.is-selected{ border-color:#111; }

/* 缩略图 */
.thumb{
  position:relative;
  width:100%;
  height: calc(var(--tile-min) * var(--thumb-ratio));
  background-size:cover; background-position:center; background-repeat:no-repeat;
  border-radius:6px; border:1px solid #e5e7eb; overflow:hidden;
}
.eye-btn{
  position:absolute; top:6px; right:6px;
  width:24px; height:24px;
  border:1px solid #e5e7eb; border-radius:6px;
  background:#fff; cursor:pointer; padding:0;
  display:flex; align-items:center; justify-content:center;
}
.eye-btn:hover{ background:#f4f4f5; }
.eye-svg{ width:16px; height:16px; fill:#9aa0a6; }
.eye-btn:hover .eye-svg{ fill:#111; }

/* 元信息行：论文来源圆点保持和 semantic / stepwise 的来源标记一致 */
.meta{
  margin-top:4px;
  font-size:11px;
  color:#111;
  display:flex;
  align-items:center;
  gap:6px;
  min-width:0;
}
.paper-dot{
  display:inline-block;
  width:10px;
  height:10px;
  border-radius:50%;
  border:1px solid rgba(255,255,255,0.25);
  background:#DCDCDC;
  flex:none;
}
.meta .title{
  display:block;
  min-width:0;
  flex:1 1 auto;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.paper-color-menu{
  position:fixed;
  z-index:10020;
  width:180px;
  box-sizing:border-box;
  padding:8px;
  border:1px solid rgba(0,0,0,.12);
  border-radius:8px;
  background:#fff;
  box-shadow:0 12px 28px rgba(15,23,42,.18);
  display:grid;
  gap:7px;
}
.paper-color-menu__row,
.paper-color-menu__actions{
  display:grid;
  align-items:center;
  gap:6px;
}
.paper-color-menu__row{
  grid-template-columns:30px 94px 26px;
}
.paper-color-menu__actions{
  grid-template-columns:1fr 1fr;
}
.paper-color-menu__input{
  width:30px;
  height:26px;
  padding:0;
  border:1px solid #d1d5db;
  border-radius:5px;
  background:#fff;
}
.paper-color-menu__hex{
  width:94px;
  min-width:94px;
  max-width:94px;
  box-sizing:border-box;
  height:26px;
  padding:0 7px;
  border:1px solid #d1d5db;
  border-radius:5px;
  background:#fff;
  color:#111827;
  font-size:11.5px;
  font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  outline:none;
}
.paper-color-menu__hex:focus{
  border-color:#111827;
  box-shadow:0 0 0 2px rgba(17,24,39,.08);
}
.paper-color-menu__random{
  width:26px;
  height:26px;
  padding:0;
  border:1px solid #111827;
  border-radius:999px;
  background:#111827;
  color:#fff;
  font-size:11px;
  font-weight:800;
  line-height:24px;
  text-align:center;
  cursor:pointer;
}
.paper-color-menu__random:hover{ background:#1f2937; }
.paper-color-menu__button{
  height:27px;
  padding:0 9px;
  border:1px solid #d1d5db;
  border-radius:6px;
  background:#fff;
  color:#111827;
  font-size:11.5px;
  cursor:pointer;
}
.paper-color-menu__button:hover{ background:#f3f4f6; }
.paper-color-menu__button_primary{
  border-color:#111827;
  background:#111827;
  color:#fff;
}
.paper-color-menu__button_primary:hover{ background:#1f2937; }
.paper-color-menu__swatches{
  display:grid;
  grid-template-columns:repeat(7, 18px);
  gap:6px;
}
.paper-color-menu__swatch{
  width:18px;
  height:18px;
  padding:0;
  border:1px solid rgba(0,0,0,.18);
  border-radius:999px;
  cursor:pointer;
}
</style>
