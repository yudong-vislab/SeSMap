# SeSMap Frontend

Vue 3 + Vite 前端:SeSMap 语义地图的可视分析界面——子空间地图、Semantic Source Gallery、Chat with LLM、Stepwise Analysis View。渲染 hex 网格上的国家(论文)与 boundary zone,支持点选/拖选/route/flight,以及每格列出该 HSU 内 MSU 的 tooltip。

## 依赖 & 启动

```bash
npm install
npm run dev        # Vite 开发服务器(默认 http://localhost:5173)
```

前端通过 Vite 代理把 `/api/*` 转发到后端(默认 `http://127.0.0.1:5000`,见 `vite.config.*` 的 `VITE_API_TARGET`)。**要先启动后端** `python3 app.py`。

```bash
npm run build      # 产出 dist/
npm run preview    # 本地预览 build 产物
```

## 主要视图与逻辑

- **Semantic Subspace Map**:从 `GET /api/semantic-map?project_id=caseN` 拉数据,按 discourse role 渲染多个子空间;每个子空间画 HSU 六边形(country=论文,boundary zone=跨论文重叠),支持 route/flight 与选择快照导出。
- **Chat with LLM**(`ChatPanel.vue`):自然语言驱动检索、子空间构造("show background & method for case3")、摘要、跨源比较;命中关键词自动切换 case。
- **Semantic Source Gallery**(`LeftPane.vue`):按 case 展示论文缩略图,并与地图国家(c0、c1…)关联上色。
- **Stepwise Analysis View**:保存 flight/HSU 为可复查分析卡片,对勾选 MSU 做结构化摘要。

## Cases 与配置

**Gallery 有两条数据源(自动择优)**:
- **方案 B(推荐,后端 manifest)**:case 若有 `data/<caseId>/gallery.json`(由后端 `extract_thumbnails.py` 生成),前端经 `GET /api/gallery?project_id=<caseId>` 拉取缩略图 + 国家/来源映射,自动灌进 gallery。**未来上传论文跑完后端流程即自动接入,无需改前端代码**(case3 走此路)。
- **旧路(bundled 资产)**:`src/assets/pictures/<folder>/*.png` + `LeftPane.vue` 里的硬编码映射(`FOLDER_ALIASES / FOLDER_PROJECT / GALLERY_PAPER_SOURCE_REGISTRY` 等)。case1/case2 走此路。

关键词别名等映射仍在 `LeftPane.vue`;`ensureBackendGallery()` 优先尝试后端 manifest,拉不到再退回 bundled 资产,两条路兼容。

> 已内置 case1(scramjet 燃烧)/ case2(时空可视化·大气)/ case3(基因组可视化)。地图与 chat 不依赖缩略图即可工作。
