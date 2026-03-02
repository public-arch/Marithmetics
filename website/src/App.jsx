import { HashRouter, Routes, Route } from 'react-router-dom';
import SiteHeader from './components/SiteHeader';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import DemoIndex from './pages/DemoIndex';
import DemoDetail from './pages/DemoDetail';
import Falsification from './pages/Falsification';
import PapersIndex from './pages/PapersIndex';
import Methodology from './pages/Methodology';
import About from './pages/About';
import Discovery from './pages/Discovery';

export default function App() {
  return (
    <HashRouter>
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <SiteHeader />
        <main style={{ flex: 1 }}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/discovery" element={<Discovery />} />
            <Route path="/demos" element={<DemoIndex />} />
            <Route path="/demos/:slug" element={<DemoDetail />} />
            <Route path="/falsification" element={<Falsification />} />
            <Route path="/papers" element={<PapersIndex />} />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </HashRouter>
  );
}
