<template>
  <header class="title-bar">
    <div class="title-brand">
      <span class="title-text">Subspace Semantic Map Visual Analytics System (SeSMapVA)</span>
    </div>
  </header>
</template>

<script setup>
</script>

<style>
.title-bar {
  height: var(--titlebar-h);
  position: relative;
  overflow: hidden;
  background:
    linear-gradient(112deg,
      #2d4f86 0%,
      #32789a 25%,
      #3f9f8a 48%,
      #c69b3f 72%,
      #bf5a7b 100%
    );
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  padding: 0 24px;
  border-bottom: 1px solid rgba(255,255,255,.18);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.18), 0 1px 8px rgba(17,24,39,.18);
  font-family: var(--app-font);
}
.title-bar::before{
  content:"";
  position:absolute;
  inset:0;
  background:
    radial-gradient(ellipse at 18% 18%, rgba(171,232,224,.24), transparent 38%),
    radial-gradient(ellipse at 52% 82%, rgba(255,227,129,.20), transparent 40%),
    radial-gradient(ellipse at 84% 24%, rgba(255,161,188,.22), transparent 38%),
    linear-gradient(180deg, rgba(255,255,255,.08) 0%, rgba(255,255,255,.035) 46%, rgba(0,0,0,.045) 100%);
  mix-blend-mode: screen;
  opacity:.72;
  pointer-events:none;
}
.title-brand{
  position: relative;
  z-index: 1;
  display:flex;
  align-items:center;
  gap:12px;
  white-space:nowrap;
}
.title-text{
  font-family: "Trebuchet MS", "Avenir Next", var(--app-font);
  font-size: 21px;
  line-height: 1;
  font-weight: 800;
  letter-spacing: .2px;
  color: transparent;
  background: linear-gradient(90deg, #ffffff 0%, #ccfff5 38%, #fff5bd 70%, #ffd4df 100%);
  -webkit-background-clip: text;
  background-clip: text;
  text-shadow: 0 2px 14px rgba(255,255,255,.2);
  filter: drop-shadow(0 2px 4px rgba(0,0,0,.18));
}
</style>
