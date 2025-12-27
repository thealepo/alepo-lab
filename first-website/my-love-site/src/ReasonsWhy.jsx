import React, { useState } from 'react';

const ReasonsWhy = ({ reasons }) => {
  const [index, setIndex] = useState(0);

  const getNewReason = () => {
    let newIndex;
    do {
      newIndex = Math.floor(Math.random() * reasons.length);
    } while (newIndex === index);
    setIndex(newIndex);
  };

  return (
    <div className="flex flex-col items-center justify-center p-10 bg-white/80 backdrop-blur-sm rounded-3xl border border-pink-100 shadow-xl max-w-lg mx-auto text-center">
      <h2 className="text-3xl font-serif text-pink-600 mb-6 font-bold">Reasons I Love You ❤️</h2>
      
      <div className="min-h-[120px] flex items-center justify-center">
        <p className="text-xl text-gray-800 leading-relaxed font-medium">
          "{reasons[index]}"
        </p>
      </div>

      <button
        onClick={getNewReason}
        className="mt-8 px-8 py-3 bg-pink-500 text-white rounded-full font-bold hover:bg-pink-600 transform transition active:scale-95 shadow-lg hover:shadow-pink-200"
      >
        Click for another reason
      </button>
    </div>
  );
};

export default ReasonsWhy;