import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./context/ThemeContext";
import { Nav } from "./components/Nav";
import { TickerTape } from "./components/TickerTape";
import { MobileTabBar } from "./components/MobileTabBar";
import { Footer } from "./components/Footer";
import { Home } from "./pages/Home";
import { Screener } from "./pages/Screener";
import { Compare } from "./pages/Compare";
import { Sectors } from "./pages/Sectors";
import { SectorDetail } from "./pages/SectorDetail";
import { CompanyDetail } from "./pages/CompanyDetail";
import { Markets } from "./pages/Markets";

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <div className="flex min-h-screen flex-col bg-background text-foreground">
          <Nav />
          <TickerTape />
          <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/screener" element={<Screener />} />
              <Route path="/markets" element={<Markets />} />
              <Route path="/markets/:tab" element={<Markets />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/sectors" element={<Sectors />} />
              <Route path="/sectors/:sector" element={<SectorDetail />} />
              <Route path="/company/:ticker" element={<CompanyDetail />} />
            </Routes>
          </main>
          <Footer />
          <MobileTabBar />
        </div>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
