import Hero from "./components/Hero";
import Benchmarks from "./components/Benchmarks";
import Crossover from "./components/Crossover";
import Honesty from "./components/Honesty";
import Explorer from "./components/Explorer";

const REPO = "https://github.com/Kart-ing/ReCompress";
const ZENODO = "https://zenodo.org/records/20786357";

export default function App() {
  return (
    <>
      <nav className="nav">
        <span className="brand">Re<span>Compress</span></span>
        <a href="#benchmarks">Benchmarks</a>
        <a href="#crossover">Crossover</a>
        <a href="#honesty">Honesty</a>
        <a href="#explorer">Explore</a>
        <span className="spacer" />
        <a href={ZENODO} target="_blank" rel="noreferrer">Paper ↗</a>
        <a href={REPO} target="_blank" rel="noreferrer">GitHub ↗</a>
      </nav>

      <Hero />
      <Benchmarks />
      <Crossover />
      <Honesty />
      <Explorer />

      <footer className="footer">
        <div>
          <a href={REPO} target="_blank" rel="noreferrer">GitHub</a>
          <a href={ZENODO} target="_blank" rel="noreferrer">Paper (Zenodo)</a>
          <a href={`${REPO}/blob/main/CITATION.cff`} target="_blank" rel="noreferrer">Cite</a>
        </div>
        <div style={{ marginTop: 8 }}>Parth Sanjay Kshirsagar · Kartikey Pandey · UC Berkeley AI Hackathon 2026</div>
        <div className="fineprint">
          All numbers replayed from real evaluation runs in <code>/results</code> — nothing synthetic.
        </div>
      </footer>
    </>
  );
}
