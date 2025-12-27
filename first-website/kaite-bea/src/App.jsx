import { useState } from 'react'
import './App.css'
import ReasonsWhy from './ReasonsWhy'
import FloatingHearts from './FloatingHearts'

function App() {
  const kaitlynReasons = [
    "You are the cutest girl in the world!",
    "You are really smart!",
    "You are so good at doing nails!",
    "You are Kaitlyn (and that's the best part)!",
    "I love you so much!",
    "You are the best girlfriend in the world!",
    "Thank you for everything you do for me!",
  ];

  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center bg-gradient-to-b from-pink-50 to-white overflow-hidden">
      
      {}
      <FloatingHearts />

      {}
      <div className="relative z-10 w-full px-4">
        <header className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-serif text-pink-700 font-bold mb-2">
            Hey Baby...
          </h1>
          <p className="text-pink-400 font-medium tracking-widest uppercase text-sm">
            A small website just for you
          </p>
        </header>

        <ReasonsWhy reasons={kaitlynReasons} />
      </div>

      {}
      <footer className="absolute bottom-8 text-pink-300 text-xs tracking-tighter uppercase font-bold">
        Made with ❤️ for Kaitlyn
      </footer>
    </div>
  )
}

export default App