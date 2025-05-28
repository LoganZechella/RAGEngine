import adapter from '@sveltejs/adapter-node';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// Using adapter-node for Docker deployment
		// See https://svelte.dev/docs/kit/adapters for more information about adapters.
		adapter: adapter({
			// Options for the Node.js adapter
			out: 'build',
			precompress: true,
			envPrefix: 'VITE_'
		})
	}
};

export default config;
