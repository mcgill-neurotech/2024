import React from 'react';
import type { SVGProps } from 'react';

const ReverseIcon = (props: SVGProps<SVGSVGElement>) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      {...props}
    >
      <path
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M3 14L14 3m-4 0h4v4m-4 10v4h4m7-11L10 21"
      ></path>
    </svg>
  );
};

export default ReverseIcon;
