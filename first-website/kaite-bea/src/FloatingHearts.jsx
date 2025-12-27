import React from 'react';
import './index.css';

const FloatingHearts = () => {

  const hearts = Array.from({ length: 15 });

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      {hearts.map((_, i) => {
        const style = {
          left: `${Math.random() * 100}%`,
          animationDuration: `${6 + Math.random() * 6}s`,
          animationDelay: `${Math.random() * 10}s`,
          fontSize: `${10 + Math.random() * 20}px`,
        };

        return (
          <span key={i} className="heart-particle" style={style}>
            ❤️
          </span>
        );
      })}
    </div>
  );
};

export default FloatingHearts;