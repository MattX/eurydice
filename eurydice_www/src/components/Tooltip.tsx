import React from "react";

export default function WithTooltip(props: TooltipProps) {
    const [showTooltip, setShowTooltip] = React.useState(false);

    return (
        <div
            className="relative grid place-content-center"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            onClick={() => setShowTooltip(!showTooltip)}
        >
            {props.children}
            {showTooltip && (
                <div className="absolute bg-gray-800 text-white text-xs rounded p-1 w-32 translate-y-10 z-10">
                    {props.text}
                </div>
            )}
        </div>
    );
}

export interface TooltipProps {
    children: React.ReactNode;
    text: string;
}