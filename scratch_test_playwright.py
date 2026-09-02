import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto('https://easydata.sbp.org.pk/api/v1/series/TS_GP_MPR_MPR.M')
        content = await page.content()
        print(content[:500])
        await browser.close()

if __name__ == '__main__':
    asyncio.run(run())
