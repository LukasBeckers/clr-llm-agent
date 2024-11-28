"use client";

import React from 'react';


interface ButtonProps {
    label?: string;
    onClick?: () => void;
    disabled?: boolean;
    className?: string;
    children?: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({label, onClick, disabled, className = '', children}) => {
    return (
        <button
        onClick={onClick}
        disabled={disabled}
        className={`px-4 py-2 rounded-md text-white font-semibold shadow-md 
        hover:bg-blue-600 bg-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed ${className}`}
        >
        {label}
    </button>
    
    );
}

export default Button; 
